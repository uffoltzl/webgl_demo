#version 330 core

uniform sampler2D diffuse_map;
uniform sampler2D shadowMap;

in vec4 FragPosLightSpace;
in vec2 frag_tex_coords;
out vec4 out_color;

in float visibility;
uniform vec4 ocean_color;

float ShadowCalculation(vec4 fragPosLightSpace)
{

    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    // check whether current frag pos is in shadow
    float shadow = currentDepth - 0.005 > closestDepth  ? 0.5 : 0.0;
    return shadow;
}


void main() {
    float shadow = ShadowCalculation(FragPosLightSpace);
    out_color = texture(diffuse_map, frag_tex_coords)*(1-shadow);

    out_color = mix(ocean_color, out_color, visibility);
}
