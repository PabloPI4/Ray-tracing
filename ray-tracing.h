#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct sphere {
    double center[3];
    double r;
} sph;

typedef struct line {
    double startPoint[3];
    double vector[3];
} ln;

typedef struct camera {
    double center[3];
    double direction[3]; /*This is the point of the space at which
			the camera is aimed (The orientation of
			the camera in the space).*/
} cam;

void readObjects(int, char **);

ln *calculateReflex(ln *ray, sph *sphere, double point[3]);

ln *calculateRay(double point1[3], double point2[3], ln *ray);

double calculateDistance(double point1[3], double point2[3]);

double scalarProduct(double vector1[3], double vector2[3]);
