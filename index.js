const scene = new THREE.Scene();
scene.background = new THREE.Color(0xdddddd)
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth / 2, window.innerHeight );
document.body.appendChild( renderer.domElement );

var loader = new OBJLoader();
loader.load('WhaleShark/WhaleShark.obj', function(obj){
	scene.add(obj);
	renderer.render(scene, camera);
});

// function animate() {
// 	requestAnimationFrame( animate );
//   cube.rotation.x += 0.01;
//   cube.rotation.y += 0.01;
// 	renderer.render( scene, camera );
// }
// animate();
