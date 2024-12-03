#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define scalarProduct(vector1, vector2) (vector1[0]*vector2[0] + vector1[1]*vector2[1] + vector1[2]*vector2[2])

typedef struct sphere {
    double center[3];
    double r;
    unsigned char color[3];
    double reflection[3];
} sph;

typedef struct plane {
    double a;
    double b;
    double c;
    double d;
    unsigned char color[3];
    double reflection[3];
} pln;

typedef struct line {
    double startPoint[3];
    double vector[3];
} ln;

typedef struct camera {
    double center[3];
    double direction[3]; /*This is the point of the space at which
			the camera is aimed (The orientation of	the camera in the space)*/
} cam;

typedef struct light {
    double center[3]; /*the position of the ligth*/
    double r;
    double itsty; /*intensity of the light*/
} lght;

typedef struct collision {
    sph *sphere;
    pln *plane;
    double angle;
    double distance;
} cllsn;

/**/
void readObjects(int, char **);

/**/
void calculateReflexSph(ln *ray, sph *sphere, double point[3]);

void calculateReflexPln(ln *ray, pln *wall, double point[3]);

/**/
void calculateRay(double point1[3], double point2[3], ln *ray);

/**/
double calculateDistance(double point1[3], double point2[3]);

double calculateAngle(ln *ray, double point[3], double centerSphere[3]);

void ray_global();

void ray_tracing(short *posScreen, int x);

void calculateCollisions(ln *, double point[3], sph **sphere, pln **wall, cllsn *collision);

int collisionWithLight(ln *ray);

void writeImage();