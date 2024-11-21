#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct sphere {
    double center[3];
    double r;
    char color[3];
    double absorption[3];
} sph;

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
    double angle;
    double distance;
} cllsn;

/**/
void readObjects(int, char **);

/**/
ln *calculateReflex(ln *ray, sph *sphere, double point[3]);

/**/
ln *calculateRay(double point1[3], double point2[3], ln *ray);

/**/
double calculateDistance(double point1[3], double point2[3]);

double calculateAngle(ln *ray, double point[3], double centerSphere[3]);

/*this function makes the scalar product between two vectors*/
double scalarProduct(double vector1[3], double vector2[3]);

void ray_global();

void ray_tracing(short *posScreen, int x);

sph *calculateCollisions(ln *, double point[3], sph *sphere, cllsn *collision);

void writeImage();