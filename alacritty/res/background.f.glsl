#if defined(GLES2_RENDERER)
#define float_t mediump float
#define color_t mediump vec4
#define FRAG_COLOR gl_FragColor
#define TEXTURE2D texture2D

varying mediump vec2 texCoords;
#else
#define float_t float
#define color_t vec4
#define TEXTURE2D texture

out vec4 FragColor;
#define FRAG_COLOR FragColor

in vec2 texCoords;
#endif

uniform sampler2D backgroundTexture;
uniform float_t backgroundOpacity;

void main() {
    color_t texel = TEXTURE2D(backgroundTexture, texCoords);

    // The image is stored with straight (non-premultiplied) alpha and is blended
    // with `SRC_ALPHA, ONE_MINUS_SRC_ALPHA`, so the opacity is applied to the
    // alpha channel only.
    FRAG_COLOR = vec4(texel.rgb, texel.a * backgroundOpacity);
}
