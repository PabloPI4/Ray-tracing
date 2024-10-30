#include <stdio.h>
#include <stdlib.h>

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
    double direction[3]; //This is the point of the space at which the camera is aimed (The orientation of the camera in the space).
} cam;

sph *spheres;
int n_spheres;
cam camera;
double light[3];

FILE* image;

void readObjects(int, char **);

int main(int argc, char *argv[]) {
    image = fopen("image.ppm","w");
    if(!image) {
	perror("fopen");
        return 3;
    }
    readObjects(argc, argv);
}

void readObjects(int argc, char *argv[]) {
    FILE *readObjects;
    double pos[3];
    double r;

    if (argc < 2) {
        readObjects = stdin;
    }
    else if (argc == 2) {
        printf("Objects are going to be readed from file named: %s\n", argv[1]);
        readObjects = fopen(argv[1], "r");
        if (readObjects == NULL) {
            fprintf(stderr, "Error opening file named: %s\n", argv[1]);
        }
    }
    else {
        fprintf(stderr, "Number of arguments incorrect. Passed %d arguments and this program only need 1\n", argc - 1);
    }

    printf("Write the position of the camera: x y z vector(x) vector(y) vector(z)\n");
    if (fscanf(readObjects, "%lf%lf%lf%lf%lf%lf", camera.center, camera.center + 1, camera.center + 2, camera.direction, camera.direction + 1, camera.direction + 2) < 6) {
        fprintf(stderr, "Error reading camera\n");
        exit(1);
    }

    printf("Write the position of the light: x y z\n");
    if (fscanf(readObjects, "%lf%lf%lf", light, light + 1, light + 2) < 3) {
        fprintf(stderr, "Error reading camera\n");
        exit(2);
    }

    int end = 0;
    while(!end) {
        printf("Reading the center and radius of the spheres: x y z r\n");
        if (fscanf(readObjects, "%lf%lf%lf%lf", pos, pos + 1, pos + 2, &r) < 4) {
            fprintf(stderr, "End of reading\n");
            end = 1;
        }
        else {
            spheres = (sph *) realloc(spheres, ++n_spheres * sizeof(struct sphere));
            for (int i = 0; i < 3; i++) {
                spheres[n_spheres - 1].center[i] = pos[i];
            }
            spheres[n_spheres - 1].r = r;
        }
    }
}
