#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

uniform mat4 model, view, projection;
uniform mat4 lightSpaceMatrix;
out vec3 w_position, w_normal;
out float visibility;
out vec4 FragPosLightSpace;
const float density = 0.015;
const float gradient = 3.0;

void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
    FragPosLightSpace =  lightSpaceMatrix * vec4(position, 1);
    w_position = vec3(view * model * vec4(position, 1.0));
    w_normal = transpose(inverse(mat3(view * model))) * normal;

    // Calculs du brouillard
    float distance = length(w_position.xyz);
    visibility = exp(-pow(distance*density, gradient));
    visibility = clamp(visibility, 0.0, 1.0);
}
