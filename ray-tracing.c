#include "ray-tracing.h"

sph *spheres;
int n_spheres;
cam camera;
double light[3];

FILE* image;

int main(int argc, char *argv[]) {
    image = fopen("image.ppm","w");
    if(!image) { /*error with the end image*/
	    perror("fopen");
        exit(3);
    }
    readObjects(argc, argv);
}

void readObjects(int argc, char *argv[]) {
    FILE *readObjects;
    double pos[3];
    double r;
    int end,i;
    char stdbol = 0; 
    if (argc < 2) {
        readObjects = stdin;
        stdbol = 1;
    }
    else if (argc == 2) {
        printf("Objects are going to be readed from the file named: %s\n", argv[1]);
        readObjects = fopen(argv[1], "r");
        if (readObjects == NULL) {
            perror("fprintf");
            fprintf(stderr, "Error opening the file named: %s\n", argv[1]);
        }
    }
    else {
        fprintf(stderr, "Incorrect number of arguments. Passed %d arguments while this program needs just 1\n", argc - 1);
    }
    if(stdbol) printf("Write the camera's position. Format: x y z vector(x) vector(y) vector(z)\n");
    if (fscanf(readObjects, "%lf%lf%lf%lf%lf%lf", camera.center, camera.center + 1, camera.center + 2, camera.direction, camera.direction + 1, camera.direction + 2) < 6) {
        fprintf(stderr, "Error reading camera\n");
        exit(1);
    }
    if(stdbol) printf("Write the light's position. Format: x y z\n");
    if (fscanf(readObjects, "%lf%lf%lf", light, light + 1, light + 2) < 3) {
        fprintf(stderr, "Error reading camera\n");
        exit(2);
    }
    end = 0;
    while(!end) {
        if(stdbol) printf("Reading the center and radius of the spheres. Format: x y z r\n");
        if (fscanf(readObjects, "%lf%lf%lf%lf", pos, pos + 1, pos + 2, &r) < 4) {
            if(stdbol) fprintf(stderr, "End of reading\n");
            end = 1;
        }
        else {
            spheres = (sph*) realloc(spheres, ++n_spheres * sizeof(sph));
            for (i = 0; i < 3; i++) {
                spheres[n_spheres - 1].center[i] = pos[i];
            }
            spheres[n_spheres - 1].r = r;
        }
    }
    if(stdbol) fclose(readObjects);
}

ln *calculateReflex(ln *ray, sph *sphere, double point[3]) {
    double vectorStdSph[3];
    double newPoint[3];
    for (int i = 0; i < 3; i++) {
        vectorStdSph[i] = (point[i] - sphere->center[i])/sphere->r;
        newPoint[i] = 2*(point[i] + vectorStdSph[i]) - (point[i] - ray->vector[i]);
    }
    return calculateRay(point, newPoint, ray);
}

ln *calculateRay(double point1[3], double point2[3], ln *ray) {
    double std;
    if (!ray) { /*ray == NULL*/
        if ((ray = (ln *) malloc(sizeof(ln))) == NULL) {
            perror("malloc");
            exit(3);
        }
    }
    for (int i = 0; i < 3; i++) {
        ray->startPoint[i] = point1[i];
        ray->vector[i] = point2[i] - point1[i];
    }
    std = scalarProduct(ray->vector, ray->vector);
    for (int i = 0; i < 3; i++) {
        ray->vector[i] /= std;
    }
    return ray;
}

double calculateDistance(double point1[3], double point2[3]) {
    double vectorDistance[3];
    for (int i = 0; i < 3; i++) {
        vectorDistance[i] = point1[i] - point2[i];
    }
    return scalarProduct(vectorDistance, vectorDistance);
}

double scalarProduct(double vector1[3], double vector2[3]) {
    return sqrt(vector1[1]*vector2[1] + vector1[2]*vector2[2] + vector1[3]*vector2[3]);
}
