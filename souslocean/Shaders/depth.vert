#version 330 core

uniform mat4 model;
uniform mat4 lightSpaceMatrix;

uniform float type;

uniform float time;
uniform float wave;

layout (location = 0) in vec3 position;

void main() {
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
  gl_Position = lightSpaceMatrix * model * vec4(newposition, 1);
}
