//! Window background image rendering.
//!
//! The background image is drawn as a single full-viewport quad right after the screen was
//! cleared and before any cell is rendered. Cells using the default background color are
//! rendered with an alpha of `0`, which makes the image visible behind the terminal content.

use std::fs::File;
use std::io::BufReader;
use std::mem;
use std::path::Path;

use log::{info, warn};
use png::{ColorType, Decoder, Transformations};

use crate::config::window::{BackgroundImage, BackgroundImageMode};
use crate::display::SizeInfo;
use crate::gl;
use crate::gl::types::*;
use crate::renderer;
use crate::renderer::shader::{ShaderProgram, ShaderVersion};

/// Shader sources for the background image program.
static BACKGROUND_SHADER_V: &str = include_str!("../../res/background.v.glsl");
static BACKGROUND_SHADER_F: &str = include_str!("../../res/background.f.glsl");

/// Decoded background image, kept on the CPU side so it can be re-uploaded after a GPU reset.
#[derive(Debug)]
pub struct BackgroundTexture {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

impl BackgroundTexture {
    /// Load and decode the configured PNG file.
    ///
    /// Any failure is reported through a warning and results in the background image being
    /// disabled, since a broken image must never take the terminal down.
    pub fn load(config: &BackgroundImage) -> Option<Self> {
        match Self::decode(&config.path) {
            Ok(image) => {
                info!(
                    "Loaded background image {:?} ({}x{})",
                    config.path, image.width, image.height
                );
                Some(image)
            },
            Err(err) => {
                warn!("Unable to load background image {:?}: {err}", config.path);
                None
            },
        }
    }

    /// Decode a PNG file into straight-alpha RGBA8.
    fn decode(path: &Path) -> Result<Self, String> {
        if path.as_os_str().is_empty() {
            return Err(String::from("no path configured"));
        }

        let file = File::open(path).map_err(|err| err.to_string())?;
        let mut decoder = Decoder::new(BufReader::new(file));

        // Expand palettes, `tRNS` chunks and sub-byte grayscale, and strip 16-bit samples.
        decoder.set_transformations(Transformations::normalize_to_color8());

        let mut reader = decoder.read_info().map_err(|err| err.to_string())?;
        let mut buffer = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buffer).map_err(|err| err.to_string())?;

        let (width, height) = (info.width, info.height);
        if width == 0 || height == 0 {
            return Err(String::from("image has a zero dimension"));
        }

        let pixels = width as usize * height as usize;
        let rgba = match info.color_type {
            ColorType::Rgba => {
                buffer.truncate(pixels * 4);
                buffer
            },
            ColorType::Rgb => {
                let mut rgba = Vec::with_capacity(pixels * 4);
                for pixel in buffer.chunks_exact(3).take(pixels) {
                    rgba.extend_from_slice(&[pixel[0], pixel[1], pixel[2], u8::MAX]);
                }
                rgba
            },
            ColorType::GrayscaleAlpha => {
                let mut rgba = Vec::with_capacity(pixels * 4);
                for pixel in buffer.chunks_exact(2).take(pixels) {
                    rgba.extend_from_slice(&[pixel[0], pixel[0], pixel[0], pixel[1]]);
                }
                rgba
            },
            ColorType::Grayscale => {
                let mut rgba = Vec::with_capacity(pixels * 4);
                for pixel in buffer.iter().take(pixels) {
                    rgba.extend_from_slice(&[*pixel, *pixel, *pixel, u8::MAX]);
                }
                rgba
            },
            color_type => {
                return Err(format!("unsupported PNG color type {color_type:?}"));
            },
        };

        if rgba.len() != pixels * 4 {
            return Err(String::from("truncated image data"));
        }

        Ok(Self { width, height, rgba })
    }
}

/// Vertex of the background quad.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct Vertex {
    /// Normalized device coordinates.
    x: f32,
    y: f32,

    /// Texture coordinates.
    u: f32,
    v: f32,
}

/// Renderer drawing the window background image.
#[derive(Debug)]
pub struct BackgroundRenderer {
    /// GL buffer objects.
    vao: GLuint,
    vbo: GLuint,

    /// Background image texture.
    texture: GLuint,

    program: BackgroundShaderProgram,

    /// Texture dimensions, required to recompute the quad on resize.
    texture_width: u32,
    texture_height: u32,

    /// Scaling mode of the image.
    mode: BackgroundImageMode,

    /// Opacity applied on top of the image's own alpha channel.
    opacity: f32,

    vertices: [Vertex; 4],
}

impl BackgroundRenderer {
    pub fn new(
        shader_version: ShaderVersion,
        image: &BackgroundTexture,
        config: &BackgroundImage,
        size_info: &SizeInfo,
    ) -> Result<Self, renderer::Error> {
        // Reject images the GPU cannot store, instead of failing the upload silently.
        let max_texture_size = unsafe {
            let mut max_texture_size: GLint = 0;
            gl::GetIntegerv(gl::MAX_TEXTURE_SIZE, &mut max_texture_size);
            max_texture_size.max(0) as u32
        };
        if max_texture_size != 0
            && (image.width > max_texture_size || image.height > max_texture_size)
        {
            return Err(renderer::Error::Other(format!(
                "background image is {}x{}, but the maximum texture size is {max_texture_size}",
                image.width, image.height
            )));
        }

        let program = BackgroundShaderProgram::new(shader_version)?;

        let mut vao: GLuint = 0;
        let mut vbo: GLuint = 0;
        let mut texture: GLuint = 0;

        unsafe {
            gl::GenVertexArrays(1, &mut vao);
            gl::GenBuffers(1, &mut vbo);

            gl::BindVertexArray(vao);
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);

            let mut attribute_offset = 0;

            // Position.
            gl::VertexAttribPointer(
                0,
                2,
                gl::FLOAT,
                gl::FALSE,
                mem::size_of::<Vertex>() as i32,
                attribute_offset as *const _,
            );
            gl::EnableVertexAttribArray(0);
            attribute_offset += mem::size_of::<f32>() * 2;

            // Texture coordinates.
            gl::VertexAttribPointer(
                1,
                2,
                gl::FLOAT,
                gl::FALSE,
                mem::size_of::<Vertex>() as i32,
                attribute_offset as *const _,
            );
            gl::EnableVertexAttribArray(1);

            // Reset buffer bindings.
            gl::BindVertexArray(0);
            gl::BindBuffer(gl::ARRAY_BUFFER, 0);

            // Upload the image.
            gl::GenTextures(1, &mut texture);
            gl::BindTexture(gl::TEXTURE_2D, texture);
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                gl::RGBA as i32,
                image.width as i32,
                image.height as i32,
                0,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                image.rgba.as_ptr() as *const _,
            );

            // Clamping avoids the NPOT restrictions of GLES2 and prevents bleeding at the edges.
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);

            gl::BindTexture(gl::TEXTURE_2D, 0);
        }

        let mut renderer = Self {
            vao,
            vbo,
            texture,
            program,
            texture_width: image.width,
            texture_height: image.height,
            mode: config.mode,
            opacity: config.opacity.as_f32(),
            vertices: Default::default(),
        };
        renderer.resize(size_info);

        Ok(renderer)
    }

    /// Recompute the quad and its texture coordinates for the new window size.
    pub fn resize(&mut self, size_info: &SizeInfo) {
        self.vertices =
            Self::compute_vertices(self.mode, self.texture_width, self.texture_height, size_info);
    }

    /// Draw the background image.
    ///
    /// The caller is responsible for setting up the viewport and the blending state, and for
    /// restoring both afterwards.
    pub fn draw(&self) {
        unsafe {
            gl::UseProgram(self.program.id());
            self.program.update_uniforms(self.opacity);

            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, self.texture);

            gl::BindVertexArray(self.vao);
            gl::BindBuffer(gl::ARRAY_BUFFER, self.vbo);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (self.vertices.len() * mem::size_of::<Vertex>()) as isize,
                self.vertices.as_ptr() as *const _,
                gl::STREAM_DRAW,
            );

            gl::DrawArrays(gl::TRIANGLE_STRIP, 0, self.vertices.len() as i32);

            // Reset all bindings we changed.
            gl::BindBuffer(gl::ARRAY_BUFFER, 0);
            gl::BindVertexArray(0);
            gl::BindTexture(gl::TEXTURE_2D, 0);
            gl::UseProgram(0);
        }
    }

    /// Compute the quad's vertices in normalized device coordinates.
    ///
    /// The quad is always centered. Modes which crop the image do so by shrinking the texture
    /// coordinates, while modes which letterbox it shrink the quad itself.
    fn compute_vertices(
        mode: BackgroundImageMode,
        texture_width: u32,
        texture_height: u32,
        size_info: &SizeInfo,
    ) -> [Vertex; 4] {
        let window_width = size_info.width().max(1.);
        let window_height = size_info.height().max(1.);
        let image_width = texture_width.max(1) as f32;
        let image_height = texture_height.max(1) as f32;

        let (quad_width, quad_height, u_size, v_size) = match mode {
            BackgroundImageMode::Stretch => (window_width, window_height, 1., 1.),
            BackgroundImageMode::Fill => {
                let scale = (window_width / image_width).max(window_height / image_height);
                let u_size = (window_width / (image_width * scale)).clamp(0., 1.);
                let v_size = (window_height / (image_height * scale)).clamp(0., 1.);
                (window_width, window_height, u_size, v_size)
            },
            BackgroundImageMode::Fit => {
                let scale = (window_width / image_width).min(window_height / image_height);
                (
                    (image_width * scale).min(window_width),
                    (image_height * scale).min(window_height),
                    1.,
                    1.,
                )
            },
            BackgroundImageMode::Center => {
                let quad_width = image_width.min(window_width);
                let quad_height = image_height.min(window_height);
                (quad_width, quad_height, quad_width / image_width, quad_height / image_height)
            },
        };

        // Center the visible part of the image.
        let u_start = (1. - u_size) / 2.;
        let v_start = (1. - v_size) / 2.;
        let u_end = u_start + u_size;
        let v_end = v_start + v_size;

        // Half extents of the centered quad in NDC.
        let x = quad_width / window_width;
        let y = quad_height / window_height;

        // Triangle strip: top-left, bottom-left, top-right, bottom-right. The first row of the
        // PNG is uploaded at `v == 0`, so it maps to the top of the window.
        [
            Vertex { x: -x, y, u: u_start, v: v_start },
            Vertex { x: -x, y: -y, u: u_start, v: v_end },
            Vertex { x, y, u: u_end, v: v_start },
            Vertex { x, y: -y, u: u_end, v: v_end },
        ]
    }
}

impl Drop for BackgroundRenderer {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteTextures(1, &self.texture);
            gl::DeleteBuffers(1, &self.vbo);
            gl::DeleteVertexArrays(1, &self.vao);
        }
    }
}

/// Background image drawing program.
#[derive(Debug)]
struct BackgroundShaderProgram {
    /// Shader program.
    program: ShaderProgram,

    /// Background image sampler.
    u_texture: Option<GLint>,

    /// Opacity of the background image.
    u_opacity: Option<GLint>,
}

impl BackgroundShaderProgram {
    fn new(shader_version: ShaderVersion) -> Result<Self, renderer::Error> {
        let program =
            ShaderProgram::new(shader_version, None, BACKGROUND_SHADER_V, BACKGROUND_SHADER_F)?;

        Ok(Self {
            u_texture: program.get_uniform_location(c"backgroundTexture").ok(),
            u_opacity: program.get_uniform_location(c"backgroundOpacity").ok(),
            program,
        })
    }

    fn id(&self) -> GLuint {
        self.program.id()
    }

    fn update_uniforms(&self, opacity: f32) {
        unsafe {
            if let Some(u_texture) = self.u_texture {
                gl::Uniform1i(u_texture, 0);
            }
            if let Some(u_opacity) = self.u_opacity {
                gl::Uniform1f(u_opacity, opacity);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Window of 800x400 pixels, with a 512x512 image.
    fn size_info() -> SizeInfo {
        SizeInfo::new(800., 400., 10., 20., 0., 0., false)
    }

    fn vertices(mode: BackgroundImageMode) -> [Vertex; 4] {
        BackgroundRenderer::compute_vertices(mode, 512, 512, &size_info())
    }

    /// The quad is a triangle strip of top-left, bottom-left, top-right and
    /// bottom-right, with `v` increasing towards the bottom of the window.
    fn assert_layout(quad: &[Vertex; 4]) {
        assert_eq!(quad[0].x, quad[1].x);
        assert_eq!(quad[2].x, quad[3].x);
        assert!(quad[0].x < quad[2].x);

        assert_eq!(quad[0].y, quad[2].y);
        assert_eq!(quad[1].y, quad[3].y);
        assert!(quad[1].y < quad[0].y);

        // The top of the window must show the top of the image (`v == v_start`).
        assert!(quad[0].v < quad[1].v);
        assert_eq!(quad[0].v, quad[2].v);
        assert!(quad[0].u < quad[2].u);

        // The quad is always centered.
        assert_eq!(quad[0].x, -quad[2].x);
        assert_eq!(quad[1].y, -quad[0].y);
    }

    #[test]
    fn stretch_covers_the_whole_window() {
        let quad = vertices(BackgroundImageMode::Stretch);
        assert_layout(&quad);

        assert_eq!(quad[0].x, -1.);
        assert_eq!(quad[0].y, 1.);
        assert_eq!(quad[0].u, 0.);
        assert_eq!(quad[0].v, 0.);
        assert_eq!(quad[3].u, 1.);
        assert_eq!(quad[3].v, 1.);
    }

    #[test]
    fn fill_crops_the_image() {
        let quad = vertices(BackgroundImageMode::Fill);
        assert_layout(&quad);

        // The quad still covers the window.
        assert_eq!(quad[0].x, -1.);
        assert_eq!(quad[0].y, 1.);

        // The image is cropped vertically, since it is scaled to the window width.
        assert_eq!(quad[0].u, 0.);
        assert_eq!(quad[3].u, 1.);
        assert_eq!(quad[0].v, 0.25);
        assert_eq!(quad[3].v, 0.75);
    }

    #[test]
    fn fit_letterboxes_the_image() {
        let quad = vertices(BackgroundImageMode::Fit);
        assert_layout(&quad);

        // A 512x512 image scaled to a height of 400 is 400 pixels wide.
        assert_eq!(quad[0].x, -0.5);
        assert_eq!(quad[0].y, 1.);

        // The whole image stays visible.
        assert_eq!(quad[0].u, 0.);
        assert_eq!(quad[0].v, 0.);
        assert_eq!(quad[3].u, 1.);
        assert_eq!(quad[3].v, 1.);
    }

    #[test]
    fn center_keeps_the_image_size() {
        let quad = vertices(BackgroundImageMode::Center);
        assert_layout(&quad);

        // 512 of 800 pixels horizontally, and cropped to the window height.
        assert_eq!(quad[0].x, -0.64);
        assert_eq!(quad[0].y, 1.);
        assert_eq!(quad[0].u, 0.);
        assert_eq!(quad[3].u, 1.);
        assert_eq!(quad[0].v, 0.109375);
        assert_eq!(quad[3].v, 0.890625);
    }

    #[test]
    fn zero_sized_image_does_not_panic() {
        let quad =
            BackgroundRenderer::compute_vertices(BackgroundImageMode::Fit, 0, 0, &size_info());
        for vertex in quad {
            assert!(vertex.x.is_finite() && vertex.y.is_finite());
            assert!(vertex.u.is_finite() && vertex.v.is_finite());
        }
    }
}
