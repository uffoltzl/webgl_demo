#version 330 core
uniform sampler2D shadowMap;
in vec3 w_position, w_normal;

// Variables liées à la lumière
uniform vec3 light_dir;
uniform mat4 view;
uniform vec3 k_d, k_a, k_s;

in vec4 FragPosLightSpace;
in float visibility;
uniform vec4 ocean_color;

out vec4 out_color;

float ShadowCalculation(vec4 fragPosLightSpace)
{
    float bias = 0.005;
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    // check whether current frag pos is in shadow
    float shadow = currentDepth - bias > closestDepth   ? 0.5 : 0.0;
    return shadow;
}
void main() {
    // Calcul de la lumière
    vec3 n = normalize(w_normal);
    vec3 l = -normalize(transpose(inverse(mat3(view))) * light_dir);
    vec3 v = normalize(- w_position);
    vec3 r = reflect(-l, n);
    float shadow = ShadowCalculation(FragPosLightSpace);
    vec3 lambert = k_d * max(dot(n, l), 0.4);
    vec3 phong = lambert + k_s * pow(max(dot(r, v), 0.4), 32.0); // k_a simuler par 0.4 dans le max
    out_color = vec4(phong, 1.0)*(1-shadow);

    // Ajout du brouillard
    out_color = mix(ocean_color, out_color, visibility);
}
