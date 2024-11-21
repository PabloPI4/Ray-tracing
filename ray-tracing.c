#include "ray-tracing.h"

sph *spheres;
int n_spheres;
cam camera;
lght light;
int width = 1920;
int height = 1080;
double vectorScreenH[3];
double vectorScreenV[3];
char *screen;
double firstPixel[3]; //This is the pixel in the lower left corner

FILE* image;

int main(int argc, char *argv[]) {
    image = fopen("image.ppm","w");
    if(!image) { /*error with the end image*/
	    perror("fopen");
        exit(3);
    }
    readObjects(argc, argv);

    screen = (short **) malloc(sizeof(short *) * width * height * 3);

    ray_global();

    free(screen);
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
    if(stdbol) printf("Write the light's position. Format: x y z intensity\n");
    if (fscanf(readObjects, "%lf%lf%lf%lf", light.pos, light.pos + 1, light.pos + 2, light.itsty) < 3) {
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

void ray_global() {
    double pointInit[3];
    cllsn collision;
    cllsn cols[20];
    int colPos = 0;
    ln ray;
    ln newRay;
    int noLight = 0;
    sph *sphereCollided = NULL;
    //CALCULAR LOS VECTORES DE LA PANTALLA

    for (int i = 0; i < width * height * 3; i+=3) {
        for (int j = 0; j < 3; j++) {
            pointInit[j] = firstPixel[j] + vectorScreenH[j] * ((i % (width * 3))/3) + vectorScreenV[i] * (i/(width * 3));
            ray.startPoint[j] = camera.center[j];
            ray.vector[j] = pointInit[j] - camera.center[j];
        }

        noLight = 0;

        while(!collisionWithLight(&ray)) {
            if ((sphereCollided = calculateCollisions(&ray, pointInit, sphereCollided, &collision)) == NULL || colPos > 19) {
                noLight = 1;
                break;
            }

            memcpy(cols + colPos, &collision, sizeof(cllsn));
            colPos++;
        }

        if (noLight) {
            screen[i] = 0;
            screen[i + 1] = 0;
            screen[i + 2] = 0;
            continue;
        }

        double distance = calculateDistance(light.pos, pointInit);
        double intensity = light.itsty / (distance * distance);
        double color[3] = {255, 255, 255};

        for (int i = colPos; i >= 0; i++) {
            color[0] *= cols[i].sphere->absorption[0];
            color[1] *= cols[i].sphere->absorption[1];
            color[2] *= cols[i].sphere->absorption[2];
            intensity *= sin(cols[i].angle);
            intensity /= cols[i].distance; 
        }

        color[0] *= (1 - 1/intensity);
    }
}

sph *calculateCollisions(ln *ray, double point[3], sph *origin, cllsn *collision) { //REVISAR ESTA FUNCIÓN
    //This function calculates the sphere that the ray collides with
    sph *sphere = NULL;
    double distance = __DBL_MAX__; /*Initialised to DBL_MAX because the first distance 
    calculated should be a candidate for the global distance. With this value we are 
    sure that every distance is lower than this one*/
    double a = scalarProduct(ray->vector, ray->vector);
    double b;
    double c;
    double insideSqrt;
    double distanceSol;
    double vector[3];
    double newFirstPoint[3];
    double returnPoint[3];

    for (int i = 0; i < n_spheres; i++) {
        if (origin != NULL && origin == spheres + i) { //If the sphere is the same as origin
            continue;
        }
        /*We calculate the values of a, b and c from the equation that calculates the 
        intersection points between a line and a sphere. More details are given in README.md*/
        b = 2*(ray->vector[0]*(ray->startPoint[0] - spheres[i].center[0]) + ray->vector[1] * 
        (ray->startPoint[1] - spheres[i].center[1]) + ray->vector[2] * (ray->startPoint[2] - spheres[i].center[2]));

        vector[0] = ray->startPoint[0] - spheres[i].center[0];
        vector[1] = ray->startPoint[1] - spheres[i].center[1];
        vector[2] = ray->startPoint[2] - spheres[i].center[2];
        c = scalarProduct(vector, vector);
        c -= (sphere->r * sphere->r);
        insideSqrt = b*b - 4*a*c;

        /*With a, b and c we calculate if the equation has any solution. We know that if 
        b^2 - 4ac < 0 the equation has no solutions*/
        if (sqrt < 0) {
            continue;
        }

        /*Then we calculate the intersection points and evaluate if they are between 
        the point from where the ray starts and infinite in the direction of the ray.
        If this happens, then we evaluate if the distance between this point and the origin 
        of the ray is lower than distances calculated with other spheres collided.
        If this happens, then the sphere collided is equal to the sphere of this iteration*/
        distanceSol = (b + sqrt(insideSqrt))/(2 * a);
        for (int j = 0; j < 2; j++) {
            int behind = 1;

            for (int j = 0; j < 3; j++) {
                newFirstPoint[i] = ray->startPoint[i] + ray->vector[i] * distanceSol;
                vector[i] = newFirstPoint[i] - point[i];

                if ((vector[i] == 0 && ray->vector[i] != 0) || vector[i]/ray->vector[i] != 1) {
                    behind = 0;
                    break;
                }
            }

            if (!behind) {
                continue;
            }

            distanceSol = calculateDistance(point, newFirstPoint);

            if (distanceSol < distance) {
                distance = distanceSol;
                sphere = spheres + i;
                returnPoint[0] = newFirstPoint[0];
                returnPoint[1] = newFirstPoint[1];
                returnPoint[2] = newFirstPoint[2];
            }

            distanceSol = (b - sqrt(insideSqrt))/(2 * a);
        }
    }

    if (sphere != NULL) {
        collision->distance = calculateDistance(point, returnPoint);
        collision->angle = calculateAngle(ray, point, sphere->center);
        collision->sphere = sphere;

        for (int i = 0; i < 3; i++) {
            point[i] = returnPoint[i];
        }
    }

    return sphere;
}

int collisionWithLight(ln *ray) {
    double a = scalarProduct(ray->vector, ray->vector);
    double b = 2*(ray->vector[0]*(ray->startPoint[0] - light.center[0]) + ray->vector[1] * 
        (ray->startPoint[1] - light.center[1]) + ray->vector[2] * (ray->startPoint[2] - light.center[2]));
    double c;
    double vector[3];

    vector[0] = ray->startPoint[0] - light.center[0];
    vector[1] = ray->startPoint[1] - light.center[1];
    vector[2] = ray->startPoint[2] - light.center[2];
    c = scalarProduct(vector, vector);
    c -= (light.r * light.r);

    if (b*b - 4*a*c < 0) {
        return 0;
    }
    else {
        return 1;
    }
}

ln *calculateReflex(ln *ray, sph *sphere, double point[3]) {
    double vectorStdSph[3];
    double newPoint[3];
    /*For calculating the reflexed ray it is necesary to calculate a second point of the line.
    For calculating this point, we calculate a point of symmetry (a point in the line that goes 
    from the center of the sphere and the intersection point of the ray and the sphere).
    Then we calculate another point of the ray, so we can calculate the symmetric point of this 
    one using the point of symmetry.
    The last step is calling calculateRay passing this 2 points (the calculated and the 
    intersection point)*/
    for (int i = 0; i < 3; i++) {
        vectorStdSph[i] = (point[i] - sphere->center[i])/sphere->r;
        newPoint[i] = 2*(point[i] + vectorStdSph[i]) - (point[i] - ray->vector[i]);
    }
    return calculateRay(point, newPoint, ray);
}

ln *calculateRay(double point1[3], double point2[3], ln *ray) {
    /*We calculate the vector that goes from point1 to point2 and then we standardizate 
    it dividing it's coordinates by the scalar product of vector*vector*/
    for (int i = 0; i < 3; i++) {
        ray->startPoint[i] = point1[i];
        ray->vector[i] = point2[i] - point1[i];
    }
    double std = sqrt(scalarProduct(ray->vector, ray->vector));
    for (int i = 0; i < 3; i++) {
        ray->vector[i] /= std;
    }
    return ray;
}

double calculateDistance(double point1[3], double point2[3]) {
    /*For calculating the distance between two points first we calculate the
    vector that goes from point1 to point2, and then we calculate the scalarProduct 
    of VectorDistance*VectorDistance*/
    double vectorDistance[3];
    for (int i = 0; i < 3; i++) {
        vectorDistance[i] = point1[i] - point2[i];
    }
    return sqrt(scalarProduct(vectorDistance, vectorDistance));
}

double calculateAngle(ln *ray, double point[3], double centerSphere[3]) {
    double vectorSphere[3];
    for (int i = 0; i < 3; i++) {
        vectorSphere[i] = point[i] - centerSphere[i];
    }

    return arccos((ray->vector[0]*vectorSphere[0] + ray->vector[1]*vectorSphere[1] + ray->vector[2]*vectorSphere[2]) / 
    (sqrt(scalarProduct(ray->vector, ray->vector)) * sqrt(scalarProduct(vectorSphere, vectorSphere))));
}

double scalarProduct(double vector1[3], double vector2[3]) {
    return vector1[0]*vector2[0] + vector1[1]*vector2[1] + vector1[2]*vector2[2];
}

void writeImage() {
    char header[20];
    sprintf(header, "P3 %d %d 255\n", width, height);
    fwrite(header, 1, 20, image);

    for (int i = 0; i < height; i++) {
        fprintf(image, " %d", (int)screen[i * width * 3]);

        for (int j = 1; j < width * 3; j++) {
            fprintf(image, " %d", screen[i * width * 3 + j]);
        }

        fprintf(image, "\n");
    }
}