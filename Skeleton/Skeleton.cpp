//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1);		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
const int nv = 50;
const float radius = 0.05;
float mousestartX=-1;
float mousestartY=-1;
bool mouseclick=false;

vec3 hypertodisk(vec2 coord)
{
	return vec3(coord.x/ sqrt(coord.x * coord.x + coord.y * coord.y + 1), coord.y/ sqrt(coord.x * coord.x + coord.y * coord.y + 1), 1);
}

vec3 disktohyper(vec2 coord) {
	return vec3(coord.x, coord.y, 1) / sqrt(1- coord.x* coord.x- coord.y* coord.y);
}

vec3 tohyper(vec2 coord)
{
	return vec3(coord.x, coord.y, sqrt(coord.x * coord.x + coord.y * coord.y + 1));
}

vec3 circleprojectioncoord(vec3 coord, vec3 edgecoord)
{
	float d = acoshf(-coord.x * edgecoord.x - coord.y * edgecoord.y + coord.z * edgecoord.z);
	vec3 v = (edgecoord - coord * coshf(d))/sinhf(d);
	vec3 result = coord * coshf(radius) + v * sinhf(radius);
	return result / result.z;
}

float hyperdistance(vec3 a, vec3 b) {
	return acoshf(-a.x * b.x - a.y * b.y + a.z * b.z);
}

vec3 hypervector(vec3 a, vec3 b, float d) {
	return (b - a * coshf(d)) / sinhf(d);
}

vec3 hyperoffset(vec3 a, vec3 v, float d) {
	return a * coshf(d) + v * sinhf(d);
}

vec3 hypertranslate(vec3 coord, vec3 mirrora, vec3 mirrorb) {
	if (!(mirrora.x == mirrorb.x && mirrora.y == mirrorb.y)) {
		float mirrorad = hyperdistance(coord, mirrora);
		vec3 va = hypervector(coord, mirrora, mirrorad);
		vec3 resulta = hyperoffset(coord, va, 2 * mirrorad);
		float mirrorbd = hyperdistance(resulta, mirrorb);
		vec3 vb = hypervector(resulta, mirrorb, mirrorbd);
		vec3 resultb = hyperoffset(resulta, vb, 2 * mirrorbd);
		float mirrord = hyperdistance(coord, resultb);
		vec3 vc = hypervector(coord, resultb, mirrord);
		return hyperoffset(coord, vc, mirrord / 2);
	}
	else
		return coord;
}

class Node {
	vec3 coord;
	vec3 newcoord;
	vec3 projection;
	unsigned int vao, vbo;
	vec3 vertices[nv];
public:
	void create() {
		float coordxseed = (float)rand() / RAND_MAX*2-1;
		float coordyseed = (float)rand() / RAND_MAX*2-1;
		vec2 twodimcoord = vec2(coordxseed, coordyseed);
		coord = tohyper(twodimcoord);
		newcoord = coord;
		projection = hypertodisk(vec2(coord.x,coord.y));
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		/*for (int i = 0; i < nv; i++) {
			float fi = i * 2 * M_PI / nv;
			vec3 circle = tohyper(vec2(cosf(fi) * radius, sinf(fi) * radius) + twodimcoord);
			vertices[i] = circleprojectioncoord(coord, circle);
		}*/
		glEnableVertexAttribArray(0);  // attribute array 0
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec3), reinterpret_cast<void*>(0));
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec3),  // # bytes
			&projection,	      	// address
			GL_STATIC_DRAW);

	}
	void draw(){
		glBindVertexArray(vao);  // Draw call
		gpuProgram.setUniform(vec3(0, 1, 0), "color");
		glPointSize(10.0f);
		glDrawArrays(GL_POINTS, 0 /*startIdx*/, 1 /*# Elements*/);
		printf("%d",glGetError());
	}

	void redraw(vec3 cX, vec3 cY) {
 		coord=hypertranslate(coord, cX, cY);
		projection = hypertodisk(vec2(coord.x, coord.y));
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		/*for (int i = 0; i < nv; i++) {
			float fi = i * 2 * M_PI / nv;
			vec3 circle = tohyper(vec2(cosf(fi) * radius, sinf(fi) * radius) + twodimcoord);
			vertices[i] = circleprojectioncoord(coord, circle);
		}*/
		glEnableVertexAttribArray(0);  // attribute array 0
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec3), reinterpret_cast<void*>(0));
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec3),  // # bytes
			&projection,	      	// address
			GL_STATIC_DRAW);
	}



	/*	float d = acoshf(-coord.x * edgecoord.x - coord.y * edgecoord.y + coord.z * edgecoord.z);
	vec3 v = (edgecoord - coord * coshf(d))/sinhf(d);
	vec3 result = coord * coshf(radius) + v * sinhf(radius);*/


	vec3 getCoord() {
		return coord;
	}

	vec3 getNewCoord() {
		return newcoord;
	}

	void setCoord(vec3 coords) {
		coord = coords;
	}
};

/*		lines[i].getStart().setCoord(lines[i].getNewStartCoord());
		lines[i].getEnd().setCoord(lines[i].getNewEndCoord());*/

class Line {
	vec3 start;
	vec3 end;
	vec3 newstartcoord;
	vec3 newendcoord;
	vec3 vertices[2];
	unsigned int vao, vbo;
public:
	void setStart(vec3 newstart) {
		start=newstart;
	}
	void setEnd(vec3 newend) {
		end=newend;
	}
	vec3 getNewStartCoord() {
		return newstartcoord;
	}
	vec3 getNewEndCoord()
	{
		return newendcoord;
	}
	void create(Node startnode, Node endnode) {

		start = startnode.getCoord();
		end = endnode.getCoord();
		newstartcoord=start;
		newendcoord=end;
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		vertices[0] = start/ start.z;
		vertices[1] = end / end.z;
		glEnableVertexAttribArray(0);  // attribute array 0
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec3), reinterpret_cast<void*>(0));
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec3) * 2,  // # bytes
			vertices,	      	// address
			GL_STATIC_DRAW);
		printf("line done\n");

	}
	void draw() {
		glBindVertexArray(vao);  // Draw call
		gpuProgram.setUniform(vec3(0, 1, 0), "color");
		glLineWidth(0.01f);
		glDrawArrays(GL_LINES, 0 /*startIdx*/, 2 /*# Elements*/);
		printf("line drawn\n");
		printf("%d", glGetError());
	}

	void redraw(vec3 cX, vec3 cY) {
		start = hypertranslate(start, cX, cY);
		end = hypertranslate(end, cX, cY);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		vertices[0] = start / start.z;
		vertices[1] = end / end.z;
		glEnableVertexAttribArray(0);  // attribute array 0
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec3), reinterpret_cast<void*>(0));
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec3) * 2,  // # bytes
			vertices,	      	// address
			GL_STATIC_DRAW);
	}
};

class Graph {
	unsigned int vao, vbo;
	std::vector<Node> controlPoints;
	std::vector<Line> neighbors;
public:
	Graph() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); 
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
		// Map attribute array 0 to the vertex data of the interleaved vbo
		 // attribute array, components/attribute, component type, normalize?, stride, offset
		// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
	}
};


unsigned int vao;	   // virtual world on the GPU
Node node[50];
std::vector<Line> lines;


// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	for (int i = 0; i < 50; i++) {
		node[i].create();
	}
	for (int i = 0; i < 50; i++)
	{
		for (int j = i; j < 50; j++)
		{
			if (rand() % 20 == 0&&i!=j) {
				lines.push_back(Line());
				lines.back().create(node[i], node[j]);
			}
		}
	}

	


	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer
	for (int i = 0; i < 50; i++)
		node[i].draw();
	for (int i = 0; i < lines.size(); i++)
		lines[i].draw();
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}



// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	if (!mouseclick) {
		mousestartX = cX;
		mousestartY = cY;
	}
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
	for (int i = 0; i < 50; i++)
		node[i].redraw(disktohyper(vec2(mousestartX,mousestartY)), disktohyper(vec2(cX, cY)));
	for (int i = 0; i < lines.size(); i++)
		lines[i].redraw(disktohyper(vec2(mousestartX, mousestartY)), disktohyper(vec2(cX, cY)));
	mouseclick = true;
	mousestartX = cX;
	mousestartY = cY;
	glutPostRedisplay();
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;


	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP: 	mouseclick = false; buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	/*for (int i = 0; i < 50; i++)
		node[i].redraw();
	glutPostRedisplay();*/
}
