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
// Nev    : Hegedus Andras
// Neptun : BCFU8E
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
const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0
    layout(location = 1) in vec2 texturevp;	// Varying input: vp = vertex position is expected in attrib array 0  
	out vec2 textureout;
    void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1);		// transform vp from modeling space to normalized device space
		textureout=texturevp;
	}
)";

// fragment shader in GLSL
const char* const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	in vec2 textureout;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel
	uniform sampler2D sampler;

	void main() {
		outColor = texture(sampler,textureout);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
const int nv = 16;
const float radius = 0.05;
float mousestartX = -1;
float mousestartY = -1;
bool mouseclick = false;

vec3 hypertodisk(vec2 coord)
{
	return vec3(coord.x / sqrt(coord.x * coord.x + coord.y * coord.y + 1), coord.y / sqrt(coord.x * coord.x + coord.y * coord.y + 1), 1);
}

vec3 disktohyper(vec2 coord) {
	return vec3(coord.x, coord.y, 1) / sqrt(1 - coord.x * coord.x - coord.y * coord.y);
}

vec3 tohyper(vec2 coord)
{
	return vec3(coord.x, coord.y, sqrt(coord.x * coord.x + coord.y * coord.y + 1));
}

vec3 circleprojectioncoord(vec3 coord, vec3 edgecoord)
{
	float d = acoshf(-coord.x * edgecoord.x - coord.y * edgecoord.y + coord.z * edgecoord.z);
	vec3 v = (edgecoord - coord * coshf(d)) / sinhf(d);
	vec3 result = coord * coshf(radius) + v * sinhf(radius);
	return result / result.z;
}

float hyperdistance(vec3 a, vec3 b) {
	return fmaxf(acoshf(-a.x * b.x - a.y * b.y + a.z * b.z), 0.001);
}

vec3 hypervector(vec3 a, vec3 b, float d) {
	return (b - a * coshf(d)) / fmaxf(sinhf(d), 0.001);
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
		vec3 result = hyperoffset(coord, vc, mirrord / 2);
		return result;
	}
	else
		return coord;
}

float distancemultiplier = 0.4 * 0.4 * 0.4 * 0.4;
float idealdistance = 0.5;

class Node {
	vec3 coord;
	vec3 newcoord;
	vec3 projection;
	unsigned int vao, vbo;
	vec3 vertices[nv];
	Node* connected[50];
	int connectedsum = 0;
	vec3 forcevec = vec3(0, 0, 0);
	vec3 velocity = vec3(0, 0, 0);
public:
	void create() {
		float coordxseed = (float)rand() / RAND_MAX * 2 - 1;
		float coordyseed = (float)rand() / RAND_MAX * 2 - 1;
		vec2 twodimcoord = vec2(coordxseed, coordyseed);
		coord = tohyper(twodimcoord);
		newcoord = coord;
		projection = hypertodisk(vec2(coord.x, coord.y));
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
	void draw() {
		glBindVertexArray(vao);  // Draw call
		glPointSize(1.0f);
		glDrawArrays(GL_POINTS, 0 /*startIdx*/, 1 /*# Elements*/);
	}

	void redraw(vec3 cX, vec3 cY) {
		coord = hypertranslate(coord, cX, cY);
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

	void addConnected(Node* node) {
		connected[connectedsum] = node;
		connectedsum++;
		printf("%d\n", connectedsum);
	}

	void calculateForce(std::vector<Node>& node) {
		forcevec = vec3(0, 0, 0);
		for (int i = 0; i < 50; i++) {

			int j = 0;
			for (j; j < connectedsum; j++)
			{
				if (&node[i] == connected[j]) {
					float distance = hyperdistance(this->getCoord(), connected[j]->getCoord());
					forcevec = forcevec + hypervector(this->getCoord(), connected[j]->getCoord(), distance) * (3 * distance - idealdistance);
					break;
				}

			}
			if (j == connectedsum)
				if (this != &node[i]) {
					float distance = hyperdistance(this->getCoord(), node[i].getCoord());
					forcevec = forcevec + hypervector(this->getCoord(), node[i].getCoord(), distance) * ((-1 / distance * distancemultiplier));
				}
		}
		float distance = hyperdistance(this->getCoord(), tohyper(vec2(0, 0)));
		forcevec = forcevec + hypervector(this->getCoord(), tohyper(vec2(0, 0)), distance) * (distance - idealdistance);
		forcevec = (forcevec * 0.01 - velocity / 8) * 15;
		/*
		while (abs(forcevec.x) < 0.01 || abs(forcevec.y) < 0.01 || abs(forcevec.z < 0.01)) {
			float a = ((float)rand() / (float)RAND_MAX * 2 - 1)/50;
			float b = ((float)rand() / (float)RAND_MAX * 2 - 1)/50;
			forcevec = tohyper(vec2(a,b));
		}*/

	}

	void forcedraw(float dt) {
		velocity = velocity + forcevec * 0.1f * dt * 400;
		coord = coord + velocity * dt * 400;
		coord = tohyper(vec2(coord.x, coord.y));
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

	vec3 getForce() {
		return forcevec;
	}

	void setForce(vec3 force) {
		forcevec = force;
	}

	int getConnectedSum() {
		return connectedsum;
	}

	bool isConnected(Node* node) {
		for (int i = 0; i < connectedsum; i++)
			if (connected[i] == node) return true;
		return false;
	}

	Node** getConnected() {
		return connected;
	}

};

/*		lines[i].getStart().setCoord(lines[i].getNewStartCoord());
		lines[i].getEnd().setCoord(lines[i].getNewEndCoord());*/

class Line {
	Node* startNode;
	Node* endNode;
	vec3 start;
	vec3 end;
	vec3 newstartcoord;
	vec3 newendcoord;
	vec3 vertices[2];
	unsigned int vao, vbo;
public:
	void setStart(vec3 newstart) {
		start = newstart;
	}
	void setEnd(vec3 newend) {
		end = newend;
	}
	vec3 getStartCoord() {
		return startNode->getCoord();
	}
	vec3 getEndCoord()
	{
		return endNode->getCoord();
	}
	void create(Node* startnode, Node* endnode) {

		startNode = startnode;
		endNode = endnode;
		start = startnode->getCoord();
		end = endnode->getCoord();
		newstartcoord = start;
		newendcoord = end;
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		vertices[0] = start / start.z;
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
		glLineWidth(0.01f);
		glDrawArrays(GL_LINES, 0 /*startIdx*/, 2 /*# Elements*/);
	}

	void redraw() {
		start = startNode->getCoord();
		end = endNode->getCoord();
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

int orientation(vec3 p1, vec3 p2, vec3 p3)
{
	// See 10th slides from following link for derivation
	// of the formula
	float val = (p2.y - p1.y) * (p3.x - p2.x) -
		(p2.x - p1.x) * (p3.y - p2.y) * 1000;
	if (val == 0) return 0;  // colinear

	return (val > 0) ? 1 : 2; // clock or counterclock wise
}

bool intersects(Line* a, Line* b) {
	vec3 p1 = hypertodisk(vec2(a->getStartCoord().x, a->getStartCoord().y));
	vec3 q1 = hypertodisk(vec2(a->getEndCoord().x, a->getEndCoord().y));
	vec3 p2 = hypertodisk(vec2(b->getStartCoord().x, b->getStartCoord().y));
	vec3 q2 = hypertodisk(vec2(b->getEndCoord().x, b->getEndCoord().y));
	int o1 = orientation(p1, q1, p2);
	int o2 = orientation(p1, q1, q2);
	int o3 = orientation(p2, q2, p1);
	int o4 = orientation(p2, q2, q1);

	// General case
	if (o1 != o2 && o3 != o4)
		return true;
	return false;

};

class Circle {
	vec3 coord;
	vec3 newcoord;
	vec3 projection;
	unsigned int vao, vbo[2];
	vec3 vertices[nv];
	vec2 texturecoords[nv];
	Node* connected[50];
	int connectedsum = 0;
	vec3 forcevec;
	Texture texture;
public:
	void create(vec4 colora, vec4 colorb) {

		coord = tohyper(vec2(0, 0));
		newcoord = coord;
		projection = hypertodisk(vec2(coord.x, coord.y));
		std::vector<vec4> textvec(128 * 128);
		for (int i = 0; i < 128; i++)
			for (int j = 0; j < 128; j++) {
				textvec[i + j * 128] = colora * ((float)i / 128) + colorb * ((128 - (float)i) / 128);
				textvec[i + j * 128].w = 1;
			}
		texture.create(128, 128, textvec, GL_NEAREST);
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(2, &vbo[0]);
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		for (int i = 0; i < nv; i++) {
			float fi = i * 2 * M_PI / nv;
			vertices[i] = hypertodisk(vec2(cosf(fi) * radius, sinf(fi) * radius));
			texturecoords[i] = vec2(cosf(fi) * 0.5 + 0.5, sinf(fi) * 0.5 + 0.5);

		}
		glEnableVertexAttribArray(0);  // attribute array 0
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec3), reinterpret_cast<void*>(0));
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec3) * nv,  // # bytes
			vertices,	      	// address
			GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		glEnableVertexAttribArray(1);  // attribute array 0
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), reinterpret_cast<void*>(0));
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * nv,  // # bytes
			texturecoords,	      	// address
			GL_STATIC_DRAW);


	}
	void draw() {

	}

	void redraw(Node node) {
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		for (int i = 0; i < nv; i++) {

			vertices[i] = disktohyper(vec2(vertices[i].x, vertices[i].y));
			vertices[i] = hypertranslate(vertices[i], coord, node.getCoord());
			vertices[i] = hypertodisk(vec2(vertices[i].x, vertices[i].y));
		}
		coord = node.getCoord();



		glEnableVertexAttribArray(0);  // attribute array 0
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec3), reinterpret_cast<void*>(0));
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec3) * nv,  // # bytes
			vertices,	      	// address
			GL_STATIC_DRAW);

		glBindVertexArray(vao);  // Draw call
		gpuProgram.setUniform(texture, "sampler");
		glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, nv /*# Elements*/);
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
std::vector<Node> node(50);
std::vector<Line> lines;
std::vector<Circle> circle(50);
std::vector<vec3> oldcoords(50);
std::vector<vec3> newcoords(50);
float oldallforce;
float newallforce;
bool init = true;

Node aa, ab, ba, bb;



// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	int maxconnected = 0;
	for (int i = 0; i < 50; i++) {
		node[i].create();
	}
	for (int j = 0; j < 50; j++)
		newcoords[j] = node[j].getCoord();
	for (int i = 0; i < 50; i++)
	{
		for (int j = i; j < 50; j++)
		{
			if (rand() % 20 == 0 && i != j) {
				lines.push_back(Line());
				lines.back().create(&node[i], &node[j]);
				node[i].addConnected(&node[j]);
				node[j].addConnected(&node[i]);
				if (maxconnected < node[i].getConnectedSum()) {
					maxconnected = node[i].getConnectedSum();

				}
			}
		}
	}


	for (int i = 0; i < 2; i++) {
		vec2 newxy[50] = { vec2(0, 0) };
		for (int j = 0; j < 50; j++) {
			for (int k = 0; k < 50; k++) {
				if (j != k) {
					if (node[j].isConnected(&node[k])) {
						newxy[j].x += node[k].getCoord().x*100/(node[j].getConnectedSum()+1);
						newxy[j].y += node[k].getCoord().y*100/ (node[j].getConnectedSum() + 1);
					}
					else {
						newxy[j].x -= node[k].getCoord().x*5/ (node[j].getConnectedSum() + 1);
						newxy[j].y -= node[k].getCoord().y*5/ (node[j].getConnectedSum() + 1);
					}
				}
			}
			newxy[j].x -= node[j].getCoord().x *10/ (node[j].getConnectedSum() + 1);
			newxy[j].y -= node[j].getCoord().y *10/ (node[j].getConnectedSum() + 1);
			newxy[j] = newxy[j] / 50;
		}
		for (int j = 0; j < 50; j++)
			node[j].setCoord(tohyper(newxy[j]));
	}
	

	/*for (int i = 0; i < 50; i++) {
		node[i].setCoord(vec3(0, 0, 1));
	}
	printf("%d", maxconnected);
	for (int i = maxconnected; i >= 0; i--) {
		int sameconnected = 0;
		float sameradius = 0.01/(float)(i+1)*(float)(maxconnected+1);
		for (int j = 0; j < 50; j++) {
			if (i == node[j].getConnectedSum())
				sameconnected++;
		}
		int k = 0;
		for (int j = 0; j < 50; j++) {
			if (i == node[j].getConnectedSum()) {
					float fi = (float)k * 2 * M_PI / (float)sameconnected;
					node[j].setCoord(tohyper(vec2(cosf(fi) * sameradius, sinf(fi) * sameradius)));
					printf("point set\n");
					k++;

			}
		}
		printf("done\n");
	}
	for (int i = 0; i < 50; i++) {
		for (int j = 0; j < node[i].getConnectedSum(); j++) {
			float fi = (float)j * 2 * M_PI / (float)node[i].getConnectedSum();
			if (node[i].getConnectedSum() > node[i].getConnected()[j]->getConnectedSum()) {
				node[i].getConnected()[j]->setCoord(tohyper(vec2(node[i].getCoord().x+cosf(fi) * 0.5, node[i].getCoord().y + sinf(fi) * 0.5)));
			}
		}
	}*/

	/*oldallforce = 10000000000000000000000.f;
	for (int i = 0; i < 10000; i++) {
		for (int j = 0; j < 50; j++) {
			float coordxseed = (float)rand() / RAND_MAX * 2 - 1;
			float coordyseed = (float)rand() / RAND_MAX * 2 - 1;
			vec2 twodimcoord = vec2(coordxseed, coordyseed);
			vec3 randcoord = tohyper(twodimcoord);
			node[j].setCoord(randcoord);
		}
		newallforce = 0;

			for (int j = 0; j < lines.size(); j++)
			{
				for (int k = j; k < lines.size(); k++)
				{
					if (intersects(&lines[j], &lines[k]) && j != k) {
						newallforce++;
						if (newallforce > oldallforce)
							break;
					}
				}
				if (newallforce > oldallforce)
					break;
			}


		printf("%f %f\n", oldallforce, newallforce);
		if (newallforce < oldallforce) {
			for (int j = 0; j < 50; j++)
				oldcoords[j] = node[j].getCoord();
			oldallforce = newallforce;
		}
	}

	for (int i = 0; i < 50; i++)
		node[i].setCoord(oldcoords[i]);

	newallforce = 0;
	for (int j = 0; j < lines.size(); j++)
	{
		for (int k = j; k < lines.size(); k++)
		{
			if (intersects(&lines[j], &lines[k])&&j!=k) {
				newallforce++;

			}
		}
	}

	printf("%f\n",newallforce);
	*/

	vec4 colors[10] = { vec4(1, 0, 0, 1),vec4(0, 1, 0, 1), vec4(0, 0, 1, 1),
					   vec4(1, 1, 0, 1),vec4(1, 0, 1, 1), vec4(0, 1, 1, 1),
					   vec4(0.5, 0, 0.5, 1),vec4(0.5, 0.5, 0, 1),vec4(0, 0.5, 0.5, 1),vec4(0.5,0.5, 0.5, 1) };
	int coloraid = 0;
	int colorbid = 0;
	for (int i = 0; i < 50; i++) {
		if (colorbid == 10) {
			coloraid++;
			colorbid = coloraid;
		}
		vec4 colora = colors[coloraid];
		vec4 colorb = colors[colorbid];
		circle[i].create(colora, colorb);
		colorbid++;
	}



	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer
	for (int i = 0; i < 50; i++){
		node[i].draw();
	}
	for (int i = 0; i < lines.size(); i++)
		lines[i].draw();
	for (int i = 0; i < 50; i++)
		node[i].redraw(disktohyper(vec2(0, 0)), disktohyper(vec2(0, 0)));
	for (int i = 0; i < lines.size(); i++)
		lines[i].redraw();
	for (int i = 0; i < 50; i++)
		circle[i].redraw(node[i]);


	glutSwapBuffers(); // exchange buffers for double buffering

}

bool physics = false;
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	if (key == ' ') {
		physics = !physics;
		glutPostRedisplay();
	}
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
		node[i].redraw(disktohyper(vec2(mousestartX, mousestartY)), disktohyper(vec2(cX, cY)));
	for (int i = 0; i < lines.size(); i++)
		lines[i].redraw();
	for (int i = 0; i < 50; i++) {
		circle[i].redraw(node[i]);
	}
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


	char* buttonStat;
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


long oldtime;
// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float dt = (time - oldtime) / 1000.f;
	oldtime = time;
	if (physics) {
		for (int i = 0; i < 50; i++) {
			node[i].calculateForce(node);

		}

		for (int i = 0; i < 50; i++) {
			node[i].forcedraw(dt);
		}

		if (length(node[49].getForce()) < 0.00001)
			physics = false;
	}


	for (int i = 0; i < lines.size(); i++)
		lines[i].redraw();
	for (int i = 0; i < 50; i++)
		circle[i].redraw(node[i]);


	glutPostRedisplay();
}