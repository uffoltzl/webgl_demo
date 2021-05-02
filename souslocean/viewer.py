#!/usr/bin/env python3
"""
Python OpenGL practical application.
"""
# Python built-in modules
import os                           # os function, i.e. checking file status
from itertools import cycle
import sys

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
import assimpcy                     # 3D resource loader
from PIL import Image               # load images for textures
from itertools import cycle
from bisect import bisect_left

from transform import *

import math
from PIL import Image

# ocean_color
ocean_color = np.array([30.0/255.0, 60.0/255.0, 90.0/255.0, 1.0])

# keyboard movement
delta = 0
RUN_SPEED = 20
TURN_SPEED = 160

SCR_WIDTH = 640
SCR_HEIGHT = 480

SHADOW_WIDTH = 1024
SHADOW_HEIGHT = 1024
light_dir = (-0.5,-1,0)
# ------------ low level OpenGL object wrappers ----------------------------
class Shader:
    """ Helper class to create and automatically destroy shader program """
    @staticmethod
    def _compile_shader(src, shader_type):
        src = open(src, 'r').read() if os.path.exists(src) else src
        src = src.decode('ascii') if isinstance(src, bytes) else src
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, src)
        GL.glCompileShader(shader)
        status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        src = ('%3d: %s' % (i+1, l) for i, l in enumerate(src.splitlines()))
        if not status:
            log = GL.glGetShaderInfoLog(shader).decode('ascii')
            GL.glDeleteShader(shader)
            src = '\n'.join(src)
            print('Compile failed for %s\n%s\n%s' % (shader_type, log, src))
            return None
        return shader

    def __init__(self, vertex_source, fragment_source):
        """ Shader can be initialized with raw strings or source file names """
        self.glid = None
        vert = self._compile_shader(vertex_source, GL.GL_VERTEX_SHADER)
        frag = self._compile_shader(fragment_source, GL.GL_FRAGMENT_SHADER)
        if vert and frag:
            self.glid = GL.glCreateProgram()  # pylint: disable=E1111
            GL.glAttachShader(self.glid, vert)
            GL.glAttachShader(self.glid, frag)
            GL.glLinkProgram(self.glid)
            GL.glDeleteShader(vert)
            GL.glDeleteShader(frag)
            status = GL.glGetProgramiv(self.glid, GL.GL_LINK_STATUS)
            if not status:
                print(GL.glGetProgramInfoLog(self.glid).decode('ascii'))
                GL.glDeleteProgram(self.glid)
                self.glid = None

    def __del__(self):
        GL.glUseProgram(0)
        if self.glid:                      # if this is a valid shader object
            GL.glDeleteProgram(self.glid)  # object dies => destroy GL object

    def use(self):
        GL.glUseProgram(self.glid)


class VertexArray:
    """ helper class to create and self destroy OpenGL vertex array objects."""
    def __init__(self, attributes, index=None, usage=GL.GL_STATIC_DRAW):
        """ Vertex array from attributes and optional index array. Vertex
            Attributes should be list of arrays with one row per vertex. """

        # create vertex array object, bind it
        self.glid = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.glid)
        self.buffers = []  # we will store buffers in a list
        nb_primitives, size = 0, 0

        # load buffer per vertex attribute (in list with index = shader layout)
        for loc, data in enumerate(attributes):
            if data is not None:
                # bind a new vbo, upload its data to GPU, declare size and type
                self.buffers.append(GL.glGenBuffers(1))
                data = np.array(data, np.float32, copy=False)  # ensure format
                nb_primitives, size = data.shape
                GL.glEnableVertexAttribArray(loc)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[-1])
                GL.glBufferData(GL.GL_ARRAY_BUFFER, data, usage)
                GL.glVertexAttribPointer(loc, size, GL.GL_FLOAT, False, 0, None)

        # optionally create and upload an index buffer for this object
        self.draw_command = GL.glDrawArrays
        self.arguments = (0, nb_primitives)
        if index is not None:
            self.buffers += [GL.glGenBuffers(1)]
            index_buffer = np.array(index, np.int32, copy=False)  # good format
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index_buffer, usage)
            self.draw_command = GL.glDrawElements
            self.arguments = (index_buffer.size, GL.GL_UNSIGNED_INT, None)

    def execute(self, primitive):
        """ draw a vertex array, either as direct array or indexed array """
        GL.glBindVertexArray(self.glid)
        self.draw_command(primitive, *self.arguments)

    def __del__(self):  # object dies => kill GL array and buffers from GPU
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(len(self.buffers), self.buffers)


# ------------  Scene object classes ------------------------------------------
class Node:
    """ Scene graph transform and parameter broadcast node """
    def __init__(self, type=1, children=(),transform=identity()):
        self.transform = transform
        self.children = list(iter(children))
        # utile pour barnabé
        self.type = type
        self.current_speed = 0
        self.current_turn_speed = 0
        # suppose qu'au début que barnabé possède une rotation nulle
        self.angle = 0

    def add(self, *drawables):
        """ Add drawables to this node, simply updating children list """
        self.children.extend(drawables)

    def draw(self, projection, view, model):
        """ Recursive draw, passing down updated model matrix. """
        for child in self.children:
            child.draw(projection, view, model @ self.transform)

    def draw_shadows(self,depthShader,model):
        # Dessine l'ombre des éléments du Node à partir du depthShader
        for child in self.children:
            child.draw_shadows(depthShader, model @ self.transform)

    def key_handler(self, key):
        """ Dispatch keyboard events to children """
        for child in self.children:
            if hasattr(child, 'key_handler'):
                child.key_handler(key)

        if self.type == 2:

            # Mouvement de Barnabé (type 2)
            # Commande clavier: les flèches
            # On le fait avancé à la vitesse RUN_SPEED
            # et on le fait pivoter à la vitesse TURN_SPEED
            if key == glfw.KEY_UP:
                self.current_speed = RUN_SPEED

            elif key == glfw.KEY_DOWN:
                self.current_speed = -RUN_SPEED
            else:
                self.current_speed = 0

            if key == glfw.KEY_RIGHT:
                self.current_turn_speed = -TURN_SPEED
            elif key == glfw.KEY_LEFT:
                self.current_turn_speed = TURN_SPEED
            else:
                self.current_turn_speed = 0

            # Calculs du mouvement effectuer
            distance = self.current_speed * delta
            self.angle += self.current_turn_speed*delta
            dx = distance * math.sin(math.radians(self.angle))
            dy = distance * math.cos(math.radians(self.angle))
            self.transform = translate((dx, 0, dy)) @ self.transform @ rotate((0,1,0), self.current_turn_speed*delta)



# --------------  Mesh class -----------------------------------
# mesh to refactor all previous classes
class Mesh:

    def __init__(self, shader, attributes, index=None):
        self.shader = shader
        names = ['view', 'projection', 'model']
        self.loc = {n: GL.glGetUniformLocation(shader.glid, n) for n in names}
        self.vertex_array = VertexArray(attributes, index)


    def draw(self,projection, view, model,primitives=GL.GL_TRIANGLES):
        GL.glUseProgram(self.shader.glid)

        GL.glUniformMatrix4fv(self.loc['view'], 1, True, view)
        GL.glUniformMatrix4fv(self.loc['projection'], 1, True, projection)
        GL.glUniformMatrix4fv(self.loc['model'], 1, True, model)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        self.vertex_array.execute(primitives)

# -------------- OpenGL Texture Wrapper ---------------------------------------
class Texture:
    """ Helper class to create and automatically destroy textures """
    def __init__(self, file, wrap_mode=GL.GL_REPEAT, min_filter=GL.GL_LINEAR,
                 mag_filter=GL.GL_LINEAR_MIPMAP_LINEAR):
        self.glid = GL.glGenTextures(1)
        try:
            # imports image as a numpy array in exactly right format
            tex = np.asarray(Image.open(file).convert('RGBA'))
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.glid)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, tex.shape[1],
                            tex.shape[0], 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, tex)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, min_filter)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, mag_filter)
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            message = 'Loaded texture %s\t(%s, %s, %s, %s)'
            print(message % (file, tex.shape, wrap_mode, min_filter, mag_filter))
        except FileNotFoundError:
            print("ERROR: unable to load texture file %s" % file)

    def __del__(self):  # delete GL texture from GPU when object dies
        GL.glDeleteTextures(self.glid)

# ------------ classe qui construit le terrain grâce à une heightmap ---------
class Terrain(Mesh):
    def __init__(self, texture,depth):
        self.size = 800
        self.texture = Texture(texture)
        self.depth = depth
        heightmap = Image.open("Terrain/heightmap.png")
        w, h = heightmap.size
        self.heightmap = heightmap.convert("L")
        self.vertex_count = h

        # hauteur maximale d'un point du terrain
        self.maxHeight = 40
        count = self.vertex_count*self.vertex_count
        vertices = [[0.0, 0.0, 0.0] for i in range(count)]
        normals = [[0.0, 0.0, 0.0] for i in range(count)]
        textureCoords = [[0.0, 0.0] for i in range(count)]
        indices = [[0, 0, 0, 0, 0, 0] for i in range(count)]
        vertexPointer = 0
        for i in range(self.vertex_count):
            for j in range(self.vertex_count):
                vertices[vertexPointer][0] = j/(self.vertex_count - 1) * self.size
                vertices[vertexPointer][1] = self.getHeight(j, i)
                vertices[vertexPointer][2] = i/(self.vertex_count - 1) * self.size
                normals[vertexPointer] = self.calculateNormal(j, i)
                textureCoords[vertexPointer][0] = j/(self.vertex_count - 1)
                textureCoords[vertexPointer][1] = i/(self.vertex_count - 1)
                vertexPointer += 1
        pointer = 0
        for gz in range(self.vertex_count):
            for gx in range(self.vertex_count):
                topLeft = (gz*self.vertex_count)+gx
                topRight = topLeft + 1
                bottomLeft = ((gz+1)*self.vertex_count)+gx
                bottomRight = bottomLeft + 1
                indices[pointer][0] = topLeft
                indices[pointer][1] = bottomLeft
                indices[pointer][2] = topRight
                indices[pointer][3] = topRight
                indices[pointer][4] = bottomLeft
                indices[pointer][5] = bottomRight
                pointer += 1

        self.attributes = [np.array(vertices), np.array(normals), np.array(textureCoords)]
        self.index = indices

        super().__init__(Shader("Shaders/terrain.vert", "Shaders/terrain.frag"), self.attributes, indices)

        # Variables de l'environnement
        loc = GL.glGetUniformLocation(self.shader.glid, 'diffuse_map')
        self.loc['diffuse_map'] = loc
        loc = GL.glGetUniformLocation(self.shader.glid, 'ocean_color')
        self.loc['ocean_color'] = loc

    def getHeight(self, x, z):
        if x < 0 or z < 0 or x >= self.vertex_count or z >= self.vertex_count:
            return 0
        else:
            height = self.heightmap.getpixel((x, z))
            max_pixel = 256
            height -= max_pixel/2
            height /= max_pixel/2
            height *= self.maxHeight
            return height

    # Calcule la normale d'un point à partir de la heightmap
    def calculateNormal(self, x, z):
        heightL = self.getHeight(x-1, z)
        heightR = self.getHeight(x+1, z)
        heightT = self.getHeight(x, z+1)
        heightB = self.getHeight(x, z-1)
        res = normalized(np.array([heightL - heightR, 2.0, heightB - heightT])).tolist()
        return res

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):
        # Transmet au shader la matrice pour passer au référentiel de la lumière
        GL.glUseProgram(self.shader.glid)
        global light_dir
        GL.glUniform1i(GL.glGetUniformLocation(self.shader.glid, "shadowMap"), 1)
        lightProjection = ortho(-60, 60, -60, 60, -60, 60)
        lightView = lookat(vec(0, 0, 0), light_dir, vec(0, 0, 1))
        lightSpaceMatrix = lightProjection @ lightView @ model
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shader.glid, "lightSpaceMatrix"), 1, True, lightSpaceMatrix)
        GL.glUniform4fv(self.loc['ocean_color'], 1, ocean_color)

        # dessine la scene normalement en utilisant la depth texture
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(self.loc['diffuse_map'], 0)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth.texture)
        super().draw(projection, view, model, primitives)

    def draw_shadows(self, depthShader, model):
        # Dessine les ombres du terrain à partir du depthShader
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(depthShader.glid, "model"), 1, True, model)
        vect_array = VertexArray(self.attributes,self.index)
        vect_array.execute(GL.GL_TRIANGLES)


# -------------- TexturedLightMesh ---------------------------------------
class TextureLightMesh(Mesh):
    def __init__(self, shader, texture, attributes, type, index=None,
                light_dir=(0, -1, 0), k_a=(0, 0, 0), k_d=(1, 1, 0), k_s=(1, 1, 1), s=16.):

        super().__init__(shader, attributes, index)

        # texture
        loc = GL.glGetUniformLocation(shader.glid, 'diffuse_map')
        self.loc['diffuse_map'] = loc
        self.texture = texture

        # Lighting
        self.light_dir = light_dir
        self.k_d, self.k_s, self.s = k_d, k_s, s
        names = ['light_dir', 's', 'k_s', 'k_d']
        loc = {n: GL.glGetUniformLocation(shader.glid, n) for n in names}
        self.loc.update(loc)

        # Fog
        self.loc['ocean_color'] = GL.glGetUniformLocation(shader.glid, "ocean_color")

        # id du poisson en fonction du type d'animation qu'on veut lui appliquer
        self.type = type
        self.loc['type'] = GL.glGetUniformLocation(shader.glid, "type")
        # Animation
        self.loc['time'] = GL.glGetUniformLocation(shader.glid, "time")
        self.loc['wave'] = GL.glGetUniformLocation(shader.glid, "wave")

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):
        GL.glUseProgram(self.shader.glid)

        # Lighting
        GL.glUniform3fv(self.loc['light_dir'], 1, self.light_dir)
        GL.glUniform3fv(self.loc['k_d'], 1, self.k_d)
        GL.glUniform3fv(self.loc['k_s'], 1, self.k_s)
        GL.glUniform1f(self.loc['s'], max(self.s, 0.001))

        # Fog
        GL.glUniform4fv(self.loc['ocean_color'], 1, ocean_color)

        # Texture
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(self.loc['diffuse_map'], 0)

        # Animation
        GL.glUniform1f(self.loc['type'], self.type)
        if self.type == 0 or self.type == 2 or self.type == 3:
            GL.glUniform1f(self.loc['time'], glfw.get_time()*2)
        if self.type == 0:
            GL.glUniform1f(self.loc['wave'], 1.0)
        elif self.type == 2:
            GL.glUniform1f(self.loc['wave'], 0.25)


        super().draw(projection, view, model, primitives)


def load_textured_light(file, shader, tex_file=None, type=1):
    """ load resources from file using assimp, return list of TexturedMesh """
    try:
        pp = assimpcy.aiPostProcessSteps
        flags = pp.aiProcess_Triangulate | pp.aiProcess_FlipUVs
        scene = assimpcy.aiImportFile(file, flags)
    except assimpcy.all.AssimpError as exception:
        print('ERROR loading', file + ': ', exception.args[0].decode())
        return []

    # Note: embedded textures not supported at the moment
    path = os.path.dirname(file) if os.path.dirname(file) != '' else './'
    for mat in scene.mMaterials:
        if not tex_file and 'TEXTURE_BASE' in mat.properties:  # texture token
            name = os.path.basename(mat.properties['TEXTURE_BASE'])
            # search texture in file's whole subdir since path often screwed up
            paths = os.walk(path, followlinks=True)
            found = [os.path.join(d, f) for d, _, n in paths for f in n
                     if name.startswith(f) or f.startswith(name)]
            assert found, 'Cannot find texture %s in %s subtree' % (name, path)
            tex_file = found[0]
        if tex_file:
            mat.properties['diffuse_map'] = Texture(file=tex_file)

    # prepare textured mesh
    meshes = []
    for mesh in scene.mMeshes:
        mat = scene.mMaterials[mesh.mMaterialIndex].properties
        assert mat['diffuse_map'], "Trying to map using a textureless material"
        attributes = [mesh.mVertices, mesh.mNormals, mesh.mTextureCoords[0]]
        mesh = TextureLightMesh(shader, mat['diffuse_map'], attributes, type, mesh.mFaces,
                    k_d=mat.get('COLOR_DIFFUSE', (0.002, 0.002, 0.002)),
                    k_s=mat.get('COLOR_SPECULAR', (0.001, 0.001, 0.001)),
                    s=mat.get('SHININESS', 1.),
                    light_dir=(0, -1, 0))
        meshes.append(mesh)

    size = sum((mesh.mNumFaces for mesh in scene.mMeshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(meshes), size))
    return meshes


# ------------ Material only ------------------------------------------------------
class PhongMesh(Mesh):
    """ Mesh with Phong illumination """

    def __init__(self, shader, attributes,depth, index=None,
                 light_dir=(0, -1, 0),   # directionnal light (in world coords)
                 k_a=(0, 0, 0), k_d=(1, 1, 0), k_s=(1, 1, 1), s=16.):
        super().__init__(shader, attributes, index)
        self.light_dir = light_dir
        self.k_a, self.k_d, self.k_s, self.s = k_a, k_d, k_s, s
        self.depth = depth

        # retrieve OpenGL locations of shader variables at initialization
        names = ['k_a', 's', 'k_s', 'k_d', 'w_camera_position']
        loc = {n: GL.glGetUniformLocation(shader.glid, n) for n in names}
        self.loc.update(loc)
        self.attributes = attributes
        self.index = index

        # Fog
        self.loc['ocean_color'] = GL.glGetUniformLocation(shader.glid, "ocean_color")

    def draw(self, projection, view, model,primitives=GL.GL_TRIANGLES):
        GL.glUseProgram(self.shader.glid)
        GL.glUniform1i(GL.glGetUniformLocation(self.shader.glid, "shadowMap"), 1)
        # setup light parameters
        GL.glUniform3fv(GL.glGetUniformLocation(self.shader.glid, "light_dir"), 1, self.light_dir)
        lightProjection = ortho(-60,60,-60,60,-60,60)
        lightView = lookat(vec(0,0,0),vec(self.light_dir), vec(0, 0, 1))
        lightSpaceMatrix = lightProjection @ lightView @ model
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shader.glid, "lightSpaceMatrix"), 1, True, lightSpaceMatrix)
        # setup material parameters
        GL.glUniform3fv(self.loc['k_a'], 1, self.k_a)
        GL.glUniform3fv(self.loc['k_d'], 1, self.k_d)
        GL.glUniform3fv(self.loc['k_s'], 1, self.k_s)
        GL.glUniform1f(self.loc['s'], max(self.s, 0.001))
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth.texture)
        # Fog
        GL.glUniform4fv(self.loc['ocean_color'], 1, ocean_color)

        super().draw(projection, view, model, primitives)

    def draw_shadows(self,depthShader,model):
        # Dessine les ombres de l'objet à partir du depthShader
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(depthShader.glid, "model"), 1, True, model)
        vect_array = VertexArray(self.attributes, self.index)
        vect_array.execute(GL.GL_TRIANGLES)

def load_phong_mesh(file, shader,depth):
    """ load resources from file using assimp, return list of ColorMesh """
    global light_dir
    try:
        pp = assimpcy.aiPostProcessSteps
        flags = pp.aiProcess_Triangulate | pp.aiProcess_GenSmoothNormals
        scene = assimpcy.aiImportFile(file, flags)
    except assimpcy.all.AssimpError as exception:
        print('ERROR loading', file + ': ', exception.args[0].decode())
        return []

    # prepare mesh nodes
    meshes = []
    for mesh in scene.mMeshes:
        mat = scene.mMaterials[mesh.mMaterialIndex].properties
        mesh = PhongMesh(shader, [mesh.mVertices, mesh.mNormals], depth, mesh.mFaces,
                         k_d=mat.get('COLOR_DIFFUSE', (1, 1, 1)),
                         k_s=mat.get('COLOR_SPECULAR', (1, 1, 1)),
                         k_a=mat.get('COLOR_AMBIENT', (0, 0, 0)),
                         s=mat.get('SHININESS', 16.),
                         light_dir=light_dir)
        meshes.append(mesh)

    size = sum((mesh.mNumFaces for mesh in scene.mMeshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(meshes), size))
    return meshes

# -------------- Depth Map ---------------------------------------
class Depth:
    """ We need to store the depth values to render the shadows into a texture,
we are using a Depth class that uses framebuffers for that """
    def __init__(self):
        #creation of the framebuffer
        self.fb = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fb)

        #create a 2D texture for the framebuffer's depth buffer
        width = 8192
        height = 8192

        self.texture=GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT, width,
        height, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D, self.texture, 0)
        GL.glDrawBuffer(GL.GL_NONE)
        GL.glReadBuffer(GL.GL_NONE)

        # Check if failed
        if not GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Bind frame buffer failed")

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)


# -------------- Shadow Mesh ---------------------------------------------------
class ShadowMesh(TextureLightMesh):
    """ Classe ShadowMesh héritée de la classe Mesh, permet d'instancier des objets avec ombre """

    def __init__(self, shader, texture, attributes, type, depth, index=None,
                light_dir=light_dir, k_a=(0, 0, 0), k_d=(1, 1, 0), k_s=(1, 1, 1), s=16.):
        super().__init__(shader, texture, attributes,type, index, light_dir, k_a, k_d , k_s, s)
        self.depth = depth
        self.light_dir = light_dir
        self.attributes =attributes
        self.index = index

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):
        # Transmet au shader la matrice pour passer au référentiel de la lumière
        GL.glUseProgram(self.shader.glid)
        GL.glUniform1i(GL.glGetUniformLocation(self.shader.glid, "shadowMap"), 1)
        lightProjection = ortho(-60,60,-60,60,-60,60)
        lightView = lookat(vec(0,0,0),vec(self.light_dir), vec(0, 0, 1))
        lightSpaceMatrix = lightProjection @ lightView @ model
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shader.glid, "lightSpaceMatrix"), 1, True, lightSpaceMatrix)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth.texture)
        super().draw(projection, view, model, primitives)

    def draw_shadows(self, depthShader, model):
        # Dessine l'ombre du mesh à partir du depthShader
        GL.glUniform1f(GL.glGetUniformLocation(depthShader.glid, "type"), self.type)
        if self.type == 0 or self.type == 2 or self.type == 3:
            GL.glUniform1f(GL.glGetUniformLocation(depthShader.glid, "time"), glfw.get_time()*2)
        if self.type == 0:
            GL.glUniform1f(GL.glGetUniformLocation(depthShader.glid, "wave"), 1.0)
        elif self.type == 2:
            GL.glUniform1f(GL.glGetUniformLocation(depthShader.glid, "wave"), 0.25)

        GL.glUniformMatrix4fv(GL.glGetUniformLocation(depthShader.glid, "model"), 1, True, model)
        vect_array = VertexArray(self.attributes,self.index)
        vect_array.execute(GL.GL_TRIANGLES)

def load_shadowed_texture(file, shader, depth, tex_file=None, type=1):
    """ load resources from file using assimp """
    global light_dir
    try:
        pp = assimpcy.aiPostProcessSteps
        flags = pp.aiProcess_Triangulate | pp.aiProcess_FlipUVs
        scene = assimpcy.aiImportFile(file, flags)
    except assimpcy.all.AssimpError as exception:
        print('ERROR loading', file + ': ', exception.args[0].decode())
        return []

    # Note: embedded textures not supported at the moment
    path = os.path.dirname(file) if os.path.dirname(file) != '' else './'
    for mat in scene.mMaterials:
        if not tex_file and 'TEXTURE_BASE' in mat.properties:  # texture token
            name = os.path.basename(mat.properties['TEXTURE_BASE'])
            # search texture in file's whole subdir since path often screwed up
            paths = os.walk(path, followlinks=True)
            found = [os.path.join(d, f) for d, _, n in paths for f in n
                     if name.startswith(f) or f.startswith(name)]
            assert found, 'Cannot find texture %s in %s subtree' % (name, path)
            tex_file = found[0]
        if tex_file:
            mat.properties['diffuse_map'] = Texture(file=tex_file)
    # prepare textured mesh
    meshes = []
    for mesh in scene.mMeshes:
        mat = scene.mMaterials[mesh.mMaterialIndex].properties
        assert mat['diffuse_map'], "Trying to map using a textureless material"
        attributes = [mesh.mVertices, mesh.mNormals, mesh.mTextureCoords[0]]
        mesh = ShadowMesh(shader, mat['diffuse_map'], attributes, type, depth, mesh.mFaces, light_dir= light_dir, k_d=mat.get('COLOR_DIFFUSE', (0.002, 0.002, 0.002)), k_s=mat.get('COLOR_SPECULAR', (0.001, 0.001, 0.001)),s=mat.get('SHININESS', 1.))
        meshes.append(mesh)

    size = sum((mesh.mNumFaces for mesh in scene.mMeshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(meshes), size))
    return meshes


# ------------  Animation -----------------------------------------------------
class KeyFrames:
    """ Stores keyframe pairs for any value type with interpolation_function"""
    def __init__(self, time_value_pairs, interpolation_function=lerp):
        if isinstance(time_value_pairs, dict):  # convert to list of pairs
            time_value_pairs = time_value_pairs.items()
        keyframes = sorted(((key[0], key[1]) for key in time_value_pairs))
        self.times, self.values = zip(*keyframes)  # pairs list -> 2 lists
        self.interpolate = interpolation_function

    def value(self, time):
        """ Computes interpolated value from keyframes, for a given time """

        # Ensure time is within bounds else return boundary keyframe
        if time < self.times[0]:
            return self.values[0]

        if self.times[-1] == self.times[0]:
            # Seulement une valeur dans les keyframes
            return self.values[-1]
        # On souhaite avoir une animation cyclique,
        # donc on revient au début quand on a fini un cycle
        newtime = time%self.times[-1]

        index = bisect_left(self.times, newtime)
        assert(index != 0)

        assert(self.times[index-1] != self.times[index])
        norm = (newtime - self.times[index-1])/(self.times[index] - self.times[index-1])
        return self.interpolate(self.values[index-1], self.values[index], norm)



class TransformKeyFrames:
    """ KeyFrames-like object dedicated to 3D transforms """
    def __init__(self, translate_keys, rotate_keys, scale_keys):
        """ stores 3 keyframe sets for translation, rotation, scale """
        self.translate = KeyFrames(translate_keys)
        self.rotate = KeyFrames(rotate_keys, quaternion_slerp)
        self.scale = KeyFrames(scale_keys)

    def value(self, time):
        """ Compute each component's interpolation and compose TRS matrix """
        T = self.translate.value(time)
        T = vec(vec(1,0,0,T[0]),vec(0,1,0,T[1]),vec(0,0,1,T[2]),vec(0,0,0,1))
        R = self.rotate.value(time)
        R = quaternion_matrix(R)
        S = self.scale.value(time)
        S = vec(vec(S,0,0,0),vec(0,S,0,0),vec(0,0,S,0),vec(0,0,0,1))
        return T @ R @ S


class KeyFrameControlNode(Node):
    """ Place node with transform keys above a controlled subtree """
    def __init__(self, translate_keys, rotate_keys, scale_keys):
        super().__init__()
        self.keyframes = TransformKeyFrames(translate_keys, rotate_keys, scale_keys)

    def draw(self, projection, view, model):
        """ When redraw requested, interpolate our node transform from keys """
        self.transform = self.keyframes.value(glfw.get_time())
        super().draw(projection, view, model)

    def draw_shadows(self,depthShader,model):
        """ When redraw requested, interpolate our node transform from keys """
        # Dessine l'ombre de l'objet à partir du depthShader
        self.transform = self.keyframes.value(glfw.get_time())
        super().draw_shadows(depthShader, model)


# ------------  Bubble  -------------------------------------------------------
class Bubble(Mesh):
    def __init__(self, nbVectrices):
        # Les vertrices forment un cercle
        # Plus ou moins précis en fonction de nbVectrices
        vectrices = [[0, 0, 0] for _ in range(nbVectrices)]
        for i in range(nbVectrices):
            vectrices[i][0] = math.cos(2*math.pi*i/nbVectrices)
            vectrices[i][1] = math.sin(2*math.pi*i/nbVectrices)
            vectrices[i][2] = 0
        super().__init__(Shader("Shaders/bubble.vert", "Shaders/bubble.frag"), [vectrices])

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):
        super().draw(projection, view, model, primitives)

    def draw_shadows(self, depthShader, model):
        # Rien à faire : une bulle n'a pas d'ombre
        return None


# ------------  Viewer class & window management ------------------------------
class GLFWTrackball(Trackball):
    """ Use in Viewer for interactive viewpoint control """

    def __init__(self, win):
        """ Init needs a GLFW window handler 'win' to register callbacks """
        super().__init__()
        self.mouse = (0, 0)
        glfw.set_cursor_pos_callback(win, self.on_mouse_move)
        glfw.set_scroll_callback(win, self.on_scroll)

    def on_mouse_move(self, win, xpos, ypos):
        """ Rotate on left-click & drag, pan on right-click & drag """
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.drag(old, self.mouse, glfw.get_window_size(win))
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.pan(old, self.mouse)

    def on_scroll(self, win, _deltax, deltay):
        """ Scroll controls the camera distance to trackball center """
        self.zoom(deltay, glfw.get_window_size(win)[1])


class Viewer(Node):
    """ GLFW viewer window, with classic initialization & graphics loop """

    def __init__(self, width=640, height=480):
        super().__init__()
        global light_dir
        self.light_dir = light_dir
        # version hints: create GL window with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)

        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)
        self.last_frame_time = glfw.get_time()

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(ocean_color[0], ocean_color[1], ocean_color[2], ocean_color [3])
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)

        # initialize trackball
        self.trackball = GLFWTrackball(self.win)

        # cyclic iterator to easily toggle polygon rendering modes
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])
        self.depth = Depth()
        self.depthShader = Shader("Shaders/depth.vert", "Shaders/depth.frag")

    def renderShadows(self, depth, light_dir):
        # Calcule et dessine les ombres des objets du Viewer
        GL.glUseProgram(self.depthShader.glid)
        lightProjection = ortho(-60, 60, -60, 60, -60, 60)
        lightView=lookat(vec(0, 0, 0), vec(light_dir), vec(0, 0, 1))
        lightSpaceMatrix =  lightProjection @ lightView
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.depthShader.glid, "lightSpaceMatrix"), 1, True, lightSpaceMatrix)
        GL.glViewport(0, 0, 8192, 8192)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, depth.fb)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        self.draw_shadows(self.depthShader, identity())
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, 640, 480)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

    def run(self):
        """ Main render loop for this OpenGL window """
        global delta
        while not glfw.window_should_close(self.win):
            # clear draw buffer and depth buffer (<-TP2)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            win_size = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(win_size)

            # Met à jour le temps écoulé
            # Nécessaire pour calculer les déplacements
            # effectués grâce au clavier
            current_frame_time = glfw.get_time()
            delta = (current_frame_time - self.last_frame_time)
            self.last_frame_time = current_frame_time
            self.renderShadows(self.depth, self.light_dir)
            # draw our scene objects
            self.draw(projection, view, identity())

            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()


    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)
            if key == glfw.KEY_W:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))

            self.key_handler(key)


# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()
    shader = Shader("Shaders/poisson.vert", "Shaders/poisson.frag")
    shaderLight = Shader("Shaders/phong.vert", "Shaders/phong.frag")

    # Création du terrain
    node_terrain = Node(transform=translate((-250,-11.7,-250)))
    node_terrain.add(Terrain("Terrain/sand.jpg",viewer.depth))
    viewer.add(node_terrain)

    # Chargement de tous les objets
    # Chaque objet est mis dans un node qui permet de définir
    # sa position, rotation et taille et aussi de se rattacher
    # à une autre entité

    barnabe_obj = "Fish/Fish/WhaleShark/WhaleShark.obj"
    barnabe_png = "Fish/Fish/WhaleShark/WhaleShark_Base_Color.png"

    hector_obj = "Fish/Fish/ReefFish5/ReefFish5.obj"
    hector_png = "Fish/Fish/ReefFish5/ReefFish5_Base_Color.png"

    susie_obj = "Fish/Fish/SeaHorse/SeaHorse.obj"
    susie_png = "Fish/Fish/SeaHorse/SeaHorse_Base_Color.png"

    edgar_obj = "Fish/Fish/BlueTang/BlueTang.obj"
    edgar_png = "Fish/Fish/BlueTang/BlueTang_Base_Color.png"

    nemo_obj = "Fish/Fish/ClownFish2/ClownFish2.obj"
    nemo_png = "Fish/Fish/ClownFish2/Clownfish2_Base_Color.png"

    caroline_obj = "Fish/Fish/Turtle/Turtle.obj"
    caroline_png = "Fish/Fish/Turtle/Turtle.jpg"

    corail_obj = "Fish/Fish/Corail/Corail.obj"
    corail_png = "Fish/Fish/Corail/Corail.jpg"

    sebastien_obj = "Fish/Fish/Crab/Crab.obj"
    sebastien_png = "Fish/Fish/Crab/Crab.jpg"

    star_obj = "Fish/Fish/BlueStarfish/BluieStarfish.obj"
    star_png = "Fish/Fish/BlueStarfish/BlueStarfish_Base_Color.png"

    cube_obj = "Fish/Fish/Cube/cube.obj"
    cube_png = "Fish/Fish/Cube/cube.png"

    suzanne_obj = "Fish/Fish/Suzanne/Suzanne.obj"

    barnabe_node = Node(2)
    meshes = load_shadowed_texture(barnabe_obj, shader, viewer.depth, barnabe_png, 2)
    for mesh in meshes:
        barnabe_node.add(mesh)

    edgar_node = Node(0, transform=translate((1.5, 0.0, 1.5)) @ scale((0.1, 0.1, 0.1)))
    meshes = load_shadowed_texture(edgar_obj, shader, viewer.depth, edgar_png, 0)
    for mesh in meshes:
        edgar_node.add(mesh)
    barnabe_node.add(edgar_node)
    viewer.add(barnabe_node)

    cube_node = Node(transform=translate((5.0,-5.0,0.0)))
    meshes = load_shadowed_texture(cube_obj, shader,viewer.depth, cube_png, 1)
    for mesh in meshes:
        cube_node.add(mesh)

    suzanne_bubble_node = Node(transform=translate((-0.4,-0.25,0.65)))
    suzanne_node = Node(transform=scale((0.25,0.25,0.25)) @ rotate((0,1,1),-30))
    meshes = load_phong_mesh(suzanne_obj,shaderLight,viewer.depth)
    for mesh in meshes:
        suzanne_node.add(mesh)
    suzanne_bubble_node.add(suzanne_node)

    bubble_translate = {0: vec(-0.15,-0.17,0.25), 3: vec(-0.2,0,0.25), 5: vec(-0.15, 0.15, 0.25), 7: vec(-0.175, 0.27, 0.25)}
    bubble_rotate = {0: quaternion()}
    bubble_scale = {0: 0.02, 7: 0.06}
    bubble_node = KeyFrameControlNode(bubble_translate, bubble_rotate, bubble_scale)
    bubble_node.add(Bubble(15))
    suzanne_bubble_node.add(bubble_node)
    cube_node.add(suzanne_bubble_node)

    susie_trans = {0: vec(1.2, 1, 0), 2: vec(1.2, 2, 0), 5: vec(1.2, 1, 0)}
    susie_scale = {0: 0.03}
    susie_rotate = {0: quaternion_from_axis_angle((0, 1, 0), degrees=-45)}
    susie_node = KeyFrameControlNode(susie_trans, susie_rotate, susie_scale)
    meshes = load_shadowed_texture(susie_obj, shader, viewer.depth, susie_png, 1)
    for mesh in meshes:
        susie_node.add(mesh)
    cube_node.add(susie_node)

    susie2_trans = {0: vec(1, 2, 0), 3: vec(1, 1.5, 0), 5: vec(1, 2, 0)}
    susie2_scale = {0: 0.05}
    susie2_rotate = {0: quaternion_from_axis_angle((0, 1, 0), degrees=45)}
    susie2_node = KeyFrameControlNode(susie2_trans, susie2_rotate, susie2_scale)
    meshes = load_shadowed_texture(susie_obj, shader, viewer.depth, susie_png, 1)
    for mesh in meshes:
        susie2_node.add(mesh)
    cube_node.add(susie2_node)

    nemo_trans = {0: vec(-25, 2, 25), 22: vec(-2, 2, 2), 40: vec(20, 2, -20)}
    nemo_scale = {0: 0.1}
    nemo_rotate = {0: quaternion_from_axis_angle((0, 1, 0), degrees=120)}
    nemo_node = KeyFrameControlNode(nemo_trans, nemo_rotate, nemo_scale)
    meshes = load_shadowed_texture(nemo_obj, shader, viewer.depth, nemo_png, 0)
    for mesh in meshes:
        nemo_node.add(mesh)
    cube_node.add(nemo_node)

    nemo_trans2 = {0: vec(-28, 2, 26), 20: vec(0, 2, 3), 40: vec(20, 2, -23)}
    nemo_scale2 = {0: 0.07}
    nemo_rotate2 = {0: quaternion_from_axis_angle((0, 1, 0), degrees=120)}
    nemo_node = KeyFrameControlNode(nemo_trans2, nemo_rotate2, nemo_scale2)
    meshes = load_shadowed_texture(nemo_obj, shader, viewer.depth, nemo_png, 0)
    for mesh in meshes:
        nemo_node.add(mesh)
    cube_node.add(nemo_node)

    nemo_trans3 = {0: vec(-22, 2, 21), 41: vec(20, 2, -20)}
    nemo_scale3 = {0: 0.07}
    nemo_rotate3 = {0: quaternion_from_axis_angle((0, 1, 0), degrees=120)}
    nemo_node = KeyFrameControlNode(nemo_trans3, nemo_rotate3, nemo_scale3)
    meshes = load_shadowed_texture(nemo_obj, shader, viewer.depth, nemo_png, 0)
    for mesh in meshes:
        nemo_node.add(mesh)
    cube_node.add(nemo_node)

    nemo_trans4 = {0: vec(-22, 2.3, 21), 39: vec(20, 2.5, -20)}
    nemo_scale4 = {0: 0.07}
    nemo_rotate4 = {0: quaternion_from_axis_angle((0, 1, 0), degrees=120)}
    nemo_node = KeyFrameControlNode(nemo_trans4, nemo_rotate4, nemo_scale4)
    meshes = load_shadowed_texture(nemo_obj, shader, viewer.depth, nemo_png, 0)
    for mesh in meshes:
        nemo_node.add(mesh)
    cube_node.add(nemo_node)

    nemo_trans5 = {0: vec(-22, 2.2, 21), 36: vec(30, 2.2, -20)}
    nemo_scale5 = {0: 0.1}
    nemo_rotate5 = {0: quaternion_from_axis_angle((0, 1, 0), degrees=120)}
    nemo_node = KeyFrameControlNode(nemo_trans5, nemo_rotate5, nemo_scale5)
    meshes = load_shadowed_texture(nemo_obj, shader, viewer.depth, nemo_png, 0)
    for mesh in meshes:
        nemo_node.add(mesh)
    cube_node.add(nemo_node)

    nemo_trans6 = {0: vec(-20, 1.7, 21), 38: vec(30, 2, -20)}
    nemo_scale6 = {0: 0.1}
    nemo_rotate6 = {0: quaternion_from_axis_angle((0, 1, 0), degrees=120)}
    nemo_node = KeyFrameControlNode(nemo_trans6, nemo_rotate6, nemo_scale6)
    meshes = load_shadowed_texture(nemo_obj, shader, viewer.depth, nemo_png, 0)
    for mesh in meshes:
        nemo_node.add(mesh)
    cube_node.add(nemo_node)

    star_node = Node(transform=translate((0,0.5,0)) @ scale((0.3, 0.3, 0.3)))
    meshes = load_shadowed_texture(star_obj, shader, viewer.depth, star_png, 1)
    for mesh in meshes:
        star_node.add(mesh)
    cube_node.add(star_node)

    translate_keys = {0: vec(1,-0.5,0.5), 10: vec(1,-0.5,-0.5), 20: vec(1,-0.5,0.5)}
    rotate_keys = {0: quaternion_mul(quaternion_from_axis_angle((1,0,0), degrees=-90), quaternion_from_axis_angle((0,0,1), degrees=90))}
    scale_keys = {0: 0.02}
    sebastien_node = KeyFrameControlNode(translate_keys, rotate_keys, scale_keys)
    meshes = load_shadowed_texture(sebastien_obj, shader, viewer.depth, sebastien_png, 1)
    for mesh in meshes:
        sebastien_node.add(mesh)
    cube_node.add(sebastien_node)
    viewer.add(cube_node)

    corail_turtle_node = Node(transform=translate((2.5, -5.0, -5.0)))
    corail_node = Node(transform=scale((0.01,0.01,0.01)))
    meshes = load_shadowed_texture(corail_obj, shader, viewer.depth, corail_png, 1)
    for mesh in meshes:
        corail_node.add(mesh)
    corail_turtle_node.add(corail_node)

    hector_trans = {0: vec(-0.5, 1, 0.5)}
    hector_scale = {0: 0.07}
    hector_rotate = {0: quaternion_from_axis_angle((0,1,0), degrees=-90), 1: quaternion_from_axis_angle((0,1,0), degrees=-180), 2: quaternion_from_axis_angle((0,1,0), degrees=-270), 3: quaternion_from_axis_angle((0,1,0), degrees=-360), 4: quaternion_from_axis_angle((0,1,0), degrees=-90)}
    hector_node = KeyFrameControlNode(hector_trans, hector_rotate, hector_scale)
    meshes = load_shadowed_texture(hector_obj, shader,viewer.depth, hector_png, 3)
    for mesh in meshes:
        hector_node.add(mesh)
    corail_turtle_node.add(hector_node)

    caroline_node = Node(transform=translate((-0.5, 0.5, 0.0)) @ scale((0.01,0.01,0.01)) @ rotate((1,0,0), 270) @ rotate((0,0,1),315) @ rotate((0,1,0), 45))
    meshes = load_shadowed_texture(caroline_obj, shader,viewer.depth,caroline_png, 0)
    for mesh in meshes:
        caroline_node.add(mesh)
    corail_turtle_node.add(caroline_node)
    viewer.add(corail_turtle_node)

    # Commande de clavier
    print("\n\n ----------------- Les commandes de clavier sont les flèches ------------------- \n\n")

    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts
