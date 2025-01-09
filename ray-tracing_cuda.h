#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define scalarProduct(vector1, vector2) (vector1[0]*vector2[0] + vector1[1]*vector2[1] + vector1[2]*vector2[2])
#define calculateAngle(vector1, vector2) (scalarProduct(vector1, vector2) / (sqrt(scalarProduct(vector1, vector1)) * sqrt(scalarProduct(vector2, vector2))))

typedef struct sphere {
    double center[3];
    double r;
    unsigned char color[3];
    int absorptionLight;
    double reflection;
} sph;

typedef struct plane {
    double coeficients[3];
    double d;
    unsigned char color[3];
    int absorptionLight;
    double reflection;
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

/**/
void readObjects(int, char **);

/**/
__device__ void calculateReflex(ln *ray, double vector[3], double point[3]);

/**/
__device__ void calculateRay(double point1[3], double point2[3], ln *ray);

/**/
__device__ double calculateDistance(double point1[3], double point2[3]);

__global__ void ray_global(cam *deviceCamera, lght *deviceLight, unsigned char *deviceScreen, sph *deviceSpheres, pln *deviceWalls, int n_spheres, int n_walls, double ambientLight, int weight, int height, double firstPixel[3], double vectorScreenH[3], double vectorScreenV[3]);

__device__ void ray_tracing(sph *deviceSpheres, pln *deviceWalls, ln *ray, double pointInit[3], double *color, int maxReflexes, double percentage, sph *sphereCollided, pln *wallCollided, lght sLight, int n_spheres, int n_walls, double ambientLight);

__device__ void calculateCollisions(sph *deviceSpheres, pln *deviceWalls, double vector[3], double point[3], int *lightRet, sph **sphere, pln **wall, lght sLight, int n_spheres, int n_walls);

void writeImage();