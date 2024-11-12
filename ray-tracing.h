#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct sphere {
    double center[3];
    double r;
    short color[3];
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
    double pos[3]; /*the position of the ligth*/
    double itsty; /*intensity of the light*/
} lght;

/**/
void readObjects(int, char **);

/**/
ln *calculateReflex(ln *ray, sph *sphere, double point[3]);

/**/
ln *calculateRay(double point1[3], double point2[3], ln *ray);

/**/
double calculateDistance(double point1[3], double point2[3]);

/*this function makes the scalar product between two vectors*/
double scalarProduct(double vector1[3], double vector2[3]);

void ray_global();

void ray_tracing(short *posScreen, int x);

sph *calculateCollisions(ln *, double point[3]);