#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
