#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 lightSpaceMatrix;
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex_position;
out vec2 frag_tex_coords;
out vec4 FragPosLightSpace;

// Variables du brouillard
out float visibility;
const float density = 0.015;
const float gradient = 3.0;

//out vec3 w_normal;

void main() {
    vec4 posToCam = view * model * vec4(position, 1.0);

    gl_Position = projection * posToCam;
    // * 40 pour que la texture soit moins étirée et donc moins flou
    frag_tex_coords = tex_position * 40;

    vec4 fragpos = model * vec4(position,1);

    FragPosLightSpace =  lightSpaceMatrix *  vec4(position,1);
    float distance = length(posToCam.xyz);
    visibility = exp(-pow(distance*density, gradient));
    visibility = clamp(visibility, 0.0, 1.0);
}
