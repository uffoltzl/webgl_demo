#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 lightSpaceMatrix;

// Variables de l'animation
uniform float type;
uniform float time;
uniform float wave;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex_position;

// Variables de la texture
out vec2 frag_tex_coords;

// Variables de la lumière
out vec3 w_position, w_normal;
out vec4 FragPosLightSpace;

// Variables du brouillard
out float visibility;
const float density = 0.015;
const float gradient = 3.0;


void main() {
    // Ajout de l'animation de l'objet
    vec3 newposition = position;
    if(type == 0){
        // Edgar frétille
        float body = (newposition.z + 1.0) / 2.0;
        newposition.x += cos(time + body) * wave;
    }
    else if (type == 2){
        // Barnabé
        float body = (newposition.z + 1.0) / 2.0;
        float mask = smoothstep(1.0, 2.0, 1.0 - body);
        newposition.x += cos(time + body) * mask * wave;
    }
    else if(type == 3) {
    // Hector tourne en rond
    newposition.x += 10 * cos(time*0.0025);
    newposition.z += 10 * sin(time*0.0025);
    }


    gl_Position = projection * view * model * vec4(newposition, 1);


    frag_tex_coords = tex_position;

    // Lumière + position dans le monde
    w_position = vec3(view * model * vec4(newposition, 1.0));
    w_normal = transpose(inverse(mat3(view * model))) * normal;

    // mat4 lightview = lookat(vec3(0),light_dir,vec3(0,1,0));

    FragPosLightSpace =  lightSpaceMatrix * vec4(newposition, 1);

    float distance = length(w_position.xyz);
    visibility = exp(-pow(distance*density, gradient));
    visibility = clamp(visibility, 0.0, 1.0);
}
