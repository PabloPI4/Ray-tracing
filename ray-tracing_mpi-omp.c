#include "ray-tracing.h"
#include <omp.h>
#include <stddef.h>
#include "mpi.h"

sph *spheres;
int n_spheres = 0;
pln *walls;
int n_walls = 0;
cam camera;
lght light;
double ambientLight = 0;
int width = 0;
int height = 0;
double vectorScreenH[3];
double vectorScreenV[3];
unsigned char *screen;
double firstPixel[3]; //This is the pixel in the lower left corner

MPI_Datatype STRUCTSPHERE;
MPI_Datatype STRUCTPLANE;
MPI_Datatype STRUCTCAMERA;
MPI_Datatype STRUCTLIGHT;

int nproc = 0, rankid = 0;
int master = 0;
int step = 0, start = 0, end = 0;

FILE* image;

void Declare_MPI_Types() {
    int lengthsCam[2] = {3,3};
    MPI_Datatype typesCam[2] = {MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint displacementsCam[2] = {offsetof(cam, center), offsetof(cam, direction)};

    int lengthsLight[3] = {3,1,1};
    MPI_Datatype typesLight[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint displacementsLight[3] = {offsetof(lght, center), offsetof(lght, r), offsetof(lght, itsty)};

    int lengthsShpere[5] = {3,1,3,1,1};
    MPI_Datatype typesShpere[5] = {MPI_DOUBLE, MPI_DOUBLE, MPI_UNSIGNED_CHAR, MPI_INT, MPI_DOUBLE};
    MPI_Aint displacementsShpere[5] = {offsetof(sph, center), offsetof(sph, r), offsetof(sph, color), offsetof(sph, absorptionLight), offsetof(sph, reflection)};

    int lengthsPlane[5] = {3,1,3,1,1};
    MPI_Datatype typesPlane[5] = {MPI_DOUBLE, MPI_DOUBLE, MPI_UNSIGNED_CHAR, MPI_INT, MPI_DOUBLE};
    MPI_Aint displacementsPlane[5] = {offsetof(pln, d), offsetof(pln, color), offsetof(pln, absorptionLight), offsetof(pln, reflection)};


    if (MPI_Type_create_struct(2, lengthsCam, displacementsCam, typesCam, &STRUCTCAMERA) != MPI_SUCCESS) {
        fprintf(stderr, "Error declaring type of struct camera\n");
        exit(22);
    }

    if (MPI_Type_create_struct(3, lengthsLight, displacementsLight, typesLight, &STRUCTLIGHT) != MPI_SUCCESS) {
        fprintf(stderr, "Error declaring type of struct light\n");
        exit(23);
    }

    if (MPI_Type_create_struct(5, lengthsShpere, displacementsShpere, typesShpere, &STRUCTSPHERE) != MPI_SUCCESS) {
        fprintf(stderr, "Error declaring type of struct sphere\n");
        exit(24);
    }

    if (MPI_Type_create_struct(5, lengthsPlane, displacementsPlane, typesPlane, &STRUCTPLANE) != MPI_SUCCESS) {
        fprintf(stderr, "Error declaring type of struct plane\n");
        exit(25);
    }

    if (MPI_Type_commit(&STRUCTCAMERA) != MPI_SUCCESS) {
        fprintf(stderr, "Error commiting struct camera\n");
        exit(30);
    }

    if (MPI_Type_commit(&STRUCTLIGHT) != MPI_SUCCESS) {
        fprintf(stderr, "Error commiting struct light\n");
        exit(31);
    }
    
    if (MPI_Type_commit(&STRUCTSPHERE) != MPI_SUCCESS) {
        fprintf(stderr, "Error commiting struct sphere\n");
        exit(32);
    }
    
    if (MPI_Type_commit(&STRUCTPLANE) != MPI_SUCCESS) {
        fprintf(stderr, "Error declaring type of struct plane\n");
        exit(33);
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankid);
    Declare_MPI_Types();

    master = 0;

    if (rankid == master) {
        image = fopen("image.ppm","w");
        if(!image) { /*error with the end image*/
	        perror("fopen");
            exit(3);
        }

        readObjects(argc, argv);
    }

    MPI_Bcast(&width, 1, MPI_INT, master, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, master, MPI_COMM_WORLD);
    MPI_Bcast(&n_spheres, 1, MPI_INT, master, MPI_COMM_WORLD);
    MPI_Bcast(&n_walls, 1, MPI_INT, master, MPI_COMM_WORLD);
    if (rankid != master) {
        spheres = (sph *) malloc(sizeof(sph) * n_spheres);
        walls = (pln *) malloc(sizeof(pln) * n_walls);
    }
    MPI_Bcast(&ambientLight, 1, MPI_DOUBLE, master, MPI_COMM_WORLD);
    MPI_Bcast(vectorScreenH, 3, MPI_DOUBLE, master, MPI_COMM_WORLD);
    MPI_Bcast(vectorScreenV, 3, MPI_DOUBLE, master, MPI_COMM_WORLD);
    MPI_Bcast(firstPixel, 3, MPI_DOUBLE, master, MPI_COMM_WORLD);
    MPI_Bcast(&camera, 1, STRUCTCAMERA, master, MPI_COMM_WORLD);
    MPI_Bcast(&light, 1, STRUCTLIGHT, master, MPI_COMM_WORLD);
    MPI_Bcast((void *)walls, n_walls, STRUCTPLANE, master, MPI_COMM_WORLD);
    MPI_Bcast((void *)spheres, n_spheres, STRUCTSPHERE, master, MPI_COMM_WORLD);

    screen = (unsigned char *) malloc(width * height * 3);

    step = (width * height)/nproc;
    if (rankid == master)
        step += (width * height)%nproc;

    step *= 3;
    start = (nproc - 1 - rankid)*(((width * height)/nproc) * 3);
    end = start + step;
    ray_global();
    if (rankid == master) {
	MPI_Status status;
	step = ((width*height)/nproc)*3;
	start = 0;
	for(int i = (nproc-1); i > 0; i--) {
		MPI_Recv((void*) &(screen[start]), step, MPI_CHAR, i, 1, MPI_COMM_WORLD, &status);
        	start += step;
	}
	writeImage();
        free(spheres);
        free(walls);
        free(screen);
    } else {
	MPI_Send((void*) &(screen[start]), step, MPI_CHAR, master, 1, MPI_COMM_WORLD);
    }
    MPI_Finalize();
}

void readObjects(int argc, char *argv[]) {
    FILE *readObjects;
    double pos[3];
    double color[3];
    double r;
    int absL;
    double reflection;
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
        exit(10);
    }

    if(stdbol) printf("Write the size of the screen: FHD (1920x1080), QHD (2560x1440) or UHD (3840x2160)\n");
    char res[4];
    if (fgets(res, 4, readObjects) == NULL) {
        fprintf(stderr, "incorrect resolution\n");
        exit(5);
    }

    if (strcmp(res, "FHD") == 0) {
        width = 5;
        height = 5;
    }
    else if (strcmp(res, "QHD") == 0) {
        width = 2560;
        height = 1440;
    }
    else if (strcmp(res, "UHD") == 0) {
        width = 3840;
        height = 2160;
    }
    else if (strcmp(res, "VLI") == 0) {
        width = 22440;
        height = 12240;
    }
    else {
        fprintf(stderr, "incorrect resolution\n");
        exit(5);
    }

    if(stdbol) printf("Write the camera's position. Format: x y z vector(x) vector(z)\n");
    if (fscanf(readObjects, "%lf%lf%lf%lf%lf", camera.center, camera.center + 1, camera.center + 2, camera.direction, camera.direction + 2) < 5) {
        fprintf(stderr, "Error reading camera\n");
        exit(1);
    }
    camera.direction[1] = 0;
    double cameraScalar = sqrt(scalarProduct(camera.direction, camera.direction));
    for (int i = 0; i < 3; i++) {
        camera.direction[i] /= cameraScalar;
    }

    if(stdbol) printf("Write the light's position. Format: x y z r\n");
    if (fscanf(readObjects, "%lf%lf%lf%lf", light.center, light.center + 1, light.center + 2, &light.r) < 4) {
        fprintf(stderr, "Error reading camera\n");
        exit(2);
    }
    light.itsty = 0.72;

    ambientLight = 0.28;

    walls = (pln *) malloc(sizeof(pln) * 6);
    n_walls = 2;

    walls[0].color[0] = 255;
    walls[0].color[1] = 255;
    walls[0].color[2] = 255;
    walls[0].absorptionLight = 800;
    walls[0].reflection = 0.8;

    if(stdbol) printf("Write the ceiling color. Format: colorR colorG colorB\n");
    if (fscanf(readObjects, "%lf%lf%lf", color, color + 1, color + 2) < 3) {
        fprintf(stderr, "Error reading ceiling color\n");
        exit(2);
    }
    walls[1].color[0] = color[0];
    walls[1].color[1] = color[1];
    walls[1].color[2] = color[2];
    walls[1].absorptionLight = -1;
    walls[1].reflection = 0.2;
    if(stdbol) printf("Write the left wall color. Format: colorR colorG colorB\n");
    if (fscanf(readObjects, "%lf%lf%lf", color, color + 1, color + 2) < 3) {
        fprintf(stderr, "Error reading left wall color\n");
        exit(2);
    }
    walls[2].color[0] = color[0];
    walls[2].color[1] = color[1];
    walls[2].color[2] = color[2];
    walls[2].absorptionLight = 80;
    walls[2].reflection = 0.2;
    if(stdbol) printf("Write the right wall color. Format: colorR colorG colorB\n");
    if (fscanf(readObjects, "%lf%lf%lf", color, color + 1, color + 2) < 3) {
        fprintf(stderr, "Error reading right wall color\n");
        exit(2);
    }
    walls[3].color[0] = color[0];
    walls[3].color[1] = color[1];
    walls[3].color[2] = color[2];
    walls[3].absorptionLight = 80;
    walls[3].reflection = 0.2;
    if(stdbol) printf("Write the front wall color. Format: colorR colorG colorB\n");
    if (fscanf(readObjects, "%lf%lf%lf", color, color + 1, color + 2) < 3) {
        fprintf(stderr, "Error reading front wall color\n");
        exit(2);
    }
    walls[4].color[0] = color[0];
    walls[4].color[1] = color[1];
    walls[4].color[2] = color[2];
    walls[4].absorptionLight = 80;
    walls[4].reflection = 0.2;
    if(stdbol) printf("Write the back wall color. Format: colorR colorG colorB\n");
    if (fscanf(readObjects, "%lf%lf%lf", color, color + 1, color + 2) < 3) {
        fprintf(stderr, "Error reading back wall color\n");
        exit(2);
    }
    walls[5].color[0] = color[0];
    walls[5].color[1] = color[1];
    walls[5].color[2] = color[2];
    walls[5].absorptionLight = 80;
    walls[5].reflection = 0.2;

    end = 0;
    while(!end) {
        if(stdbol) printf("Reading the center, radius and color of the spheres. Format: x y z r colorR colorG colorB reflectionInteger\n");
        if (fscanf(readObjects, "%lf%lf%lf%lf%lf%lf%lf%d%lf", pos, pos + 1, pos + 2, &r, color + 0, color + 1, color + 2, &absL, &reflection) < 7) {
            if(stdbol) fprintf(stderr, "End of reading\n");
            end = 1;
        }
        else {
            n_spheres++;
            if ((spheres = (sph*) realloc(spheres, n_spheres * sizeof(sph))) == NULL) {
                fprintf(stderr, "Error in realloc spheres\n");
                exit(80);
            }
            for (i = 0; i < 3; i++) {
                spheres[n_spheres - 1].center[i] = pos[i];
                spheres[n_spheres - 1].color[i] = color[i];
            }
            spheres[n_spheres - 1].r = r;
            spheres[n_spheres - 1].absorptionLight = absL;
            spheres[n_spheres - 1].reflection = reflection;
        }
    }

    double centerPoint[3];
    centerPoint[0] = camera.center[0] + camera.direction[0];
    centerPoint[1] = camera.center[1] + camera.direction[1];
    centerPoint[2] = camera.center[2] + camera.direction[2];

    vectorScreenV[0] = 0;
    vectorScreenV[1] = 25.6/width;
    vectorScreenV[2] = 0;

    vectorScreenH[0] = camera.direction[1] * vectorScreenV[2] - camera.direction[2] * vectorScreenV[1];
    vectorScreenH[1] = camera.direction[2] * vectorScreenV[0] - camera.direction[0] * vectorScreenV[2];
    vectorScreenH[2] = camera.direction[0] * vectorScreenV[1] - camera.direction[1] * vectorScreenV[0];

    firstPixel[0] = centerPoint[0] - vectorScreenH[0]*(width/2) - vectorScreenV[0]*(height/2);
    firstPixel[1] = centerPoint[1] - vectorScreenH[1]*(width/2) - vectorScreenV[1]*(height/2);
    firstPixel[2] = centerPoint[2] - vectorScreenH[2]*(width/2) - vectorScreenV[2]*(height/2);

    walls[0].coeficients[0] = -vectorScreenV[0];
    walls[0].coeficients[1] = -vectorScreenV[1];
    walls[0].coeficients[2] = -vectorScreenV[2];
    walls[0].d = -(walls[0].coeficients[0] + walls[0].coeficients[1] * (centerPoint[1] + vectorScreenV[1] * (height/2)));

    walls[4].coeficients[0] = vectorScreenV[0];
    walls[4].coeficients[1] = vectorScreenV[1];
    walls[4].coeficients[2] = vectorScreenV[2];
    walls[4].d = -(walls[4].coeficients[0] + walls[4].coeficients[1] * (centerPoint[1] - vectorScreenV[1] * (height/2)));

    walls[2].coeficients[0] = -vectorScreenH[0];
    walls[2].coeficients[1] = -vectorScreenH[1];
    walls[2].coeficients[2] = -vectorScreenH[2];
    walls[2].d = -(walls[2].coeficients[0] + walls[2].coeficients[2] * (centerPoint[2] + vectorScreenH[2] * (width/2)));

    walls[3].coeficients[0] = vectorScreenH[0];
    walls[3].coeficients[1] = vectorScreenH[1];
    walls[3].coeficients[2] = vectorScreenH[2];
    walls[3].d = -(walls[3].coeficients[0] + walls[3].coeficients[2] * (centerPoint[2] - vectorScreenH[2] * (width/2)));

    walls[1].coeficients[0] = -camera.direction[0];
    walls[1].coeficients[1] = -camera.direction[1];
    walls[1].coeficients[2] = -camera.direction[2];
    walls[1].d = -(walls[1].coeficients[0] * (centerPoint[0] + camera.direction[0] * 16));

    walls[5].coeficients[0] = camera.direction[0];
    walls[5].coeficients[1] = camera.direction[1];
    walls[5].coeficients[2] = camera.direction[2];
    walls[5].d = -(walls[5].coeficients[0] * (centerPoint[0] - camera.direction[0] * 7));

    if(stdbol) fclose(readObjects);
}

void ray_global() {
    double color[3];
    int maxReflexes = 2;
    double pointInit[3];
    ln ray;
    #pragma omp parallel for private(color, pointInit, ray) schedule(runtime)
    for (int i = start; i < end; i+=3) {
        for (int j = 0; j < 3; j++) {
            pointInit[j] = firstPixel[j] + vectorScreenH[j] * ((i % (width * 3))/3) + vectorScreenV[j] * (i/(width * 3));
            ray.startPoint[j] = pointInit[j];
            ray.vector[j] = pointInit[j] - camera.center[j];
            color[j] = 0;
        }
	ray_tracing(&ray, pointInit, color, maxReflexes, 1, NULL, NULL);
	if (color[0] > 255) {
            screen[i] = 255;
        }
        else {
            screen[i] = (unsigned char) color[0];
        }
        if (color[1] > 255) {
            screen[i + 1] = 255;
        }
        else {
            screen[i + 1] = (unsigned char) color[1];
        }
        if (color[2] > 255) {
            screen[i + 2] = 255;
        }
        else {
            screen[i + 2] = (unsigned char) color[2];
        }
    }
}

void ray_tracing(ln *ray, double pointInit[3], double *color, int maxReflexes, double percentage, sph *sphereCollided, pln *wallCollided) {
    int lightRet;
    double vector[3];
    double intensity = 0;
    sph *sphere = NULL;
    pln *wall = NULL;

    calculateCollisions(ray->vector, pointInit, &lightRet, &sphereCollided, &wallCollided);

    if (!lightRet && !sphereCollided && !wallCollided) {
        color[0] = 0;
        color[1] = 0;
        color[2] = 0;
    }
    else if (!lightRet) {
        double vectorObject[3];
        int absL;
        double reflec;
        double col[3];
        ln rayL;

        rayL.vector[0] = light.center[0] - pointInit[0];
        rayL.vector[1] = light.center[1] - pointInit[1];
        rayL.vector[2] = light.center[2] - pointInit[2];
        double std = scalarProduct(rayL.vector, rayL.vector);
        rayL.vector[0] /= std;
        rayL.vector[1] /= std;
        rayL.vector[2] /= std;
        rayL.startPoint[0] = pointInit[0] + rayL.vector[0] * 0.0005;
        rayL.startPoint[1] = pointInit[1] + rayL.vector[1] * 0.0005;
        rayL.startPoint[2] = pointInit[2] + rayL.vector[2] * 0.0005;

        if (wallCollided == NULL) {
            vectorObject[0] = pointInit[0] - sphereCollided->center[0];
            vectorObject[1] = pointInit[1] - sphereCollided->center[1];
            vectorObject[2] = pointInit[2] - sphereCollided->center[2];
            absL = sphereCollided->absorptionLight;
            reflec = sphereCollided->reflection;
            col[0] = sphereCollided->color[0];
            col[1] = sphereCollided->color[1];
            col[2] = sphereCollided->color[2];
        }
        else {
            vectorObject[0] = wallCollided->coeficients[0];
            vectorObject[1] = wallCollided->coeficients[1];
            vectorObject[2] = wallCollided->coeficients[2];
            absL = wallCollided->absorptionLight;
            reflec = wallCollided->reflection;
            col[0] = wallCollided->color[0];
            col[1] = wallCollided->color[1];
            col[2] = wallCollided->color[2];
        }

        calculateCollisions(rayL.vector, rayL.startPoint, &lightRet, &sphere, &wall);

        intensity += ambientLight;

        if (sphere == NULL && wall == NULL) {
            intensity += light.itsty * calculateAngle(rayL.vector, vectorObject);

            if (absL >= 0) {
                calculateReflex(&rayL, vectorObject, pointInit);
                intensity += light.itsty * pow(calculateAngle(rayL.vector, ray->vector), absL);
            }
        }

        if (maxReflexes > 0 && absL >= 0) {
            calculateReflex(ray, vectorObject, pointInit);
            ray_tracing(ray, pointInit, color, maxReflexes - 1, reflec, sphereCollided, wallCollided);
        }

        color[0] = (color[0] + col[0] * (1 - reflec) * intensity) * percentage;
        color[1] = (color[1] + col[1] * (1 - reflec) * intensity) * percentage;
        color[2] = (color[2] + col[2] * (1 - reflec) * intensity) * percentage;
    }
    else {
        color[0] = 255 * percentage;
        color[1] = 255 * percentage;
        color[2] = 255 * percentage;
    }
}

void calculateCollisions(double vectorRay[3], double point[3], int *lightRet, sph **originSph, pln **originWall) { //REVISAR ESTA FUNCIÓN
    //This function calculates the sphere that the ray collides with
    sph *sphere = NULL;
    pln *plane = NULL;
    *lightRet = 0;
    double distance = __DBL_MAX__; /*Initialised to DBL_MAX because the first distance 
    calculated should be a candidate for the global distance. With this value we are 
    sure that every distance is lower than this one*/
    double a = scalarProduct(vectorRay, vectorRay);
    double b;
    double c;
    double insideSqrt;
    double distanceSol;
    double vector[3];
    double newFirstPoint[3];
    double returnPoint[3];

    for (int i = 0; i < n_spheres; i++) {
        if (*originSph != NULL && *originSph == spheres + i) { //If the sphere is the same as origin
            continue;
        }
        /*We calculate the values of a, b and c from the equation that calculates the 
        intersection points between a line and a sphere. More details are given in README.md*/
        vector[0] = point[0] - spheres[i].center[0];
        vector[1] = point[1] - spheres[i].center[1];
        vector[2] = point[2] - spheres[i].center[2];
        b = -2*(vectorRay[0]*vector[0] + vectorRay[1]*vector[1] + vectorRay[2]*vector[2]);
        c = scalarProduct(vector, vector);
        c -= (spheres[i].r * spheres[i].r);
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
            int behind = 0;

            for (int k = 0; k < 3; k++) {
                newFirstPoint[k] = point[k] + vectorRay[k] * distanceSol;
                vector[k] = newFirstPoint[k] - point[k];

                if (vector[k]/vectorRay[k] < 0) {
                    behind = 1;
                    break;
                }
            }

            if (behind) {
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

    for (int i = 0; i < n_walls; i++) {
        int behind = 0;

        if (*originWall != NULL && *originWall == walls + i) {
            continue;
        }

        insideSqrt = vectorRay[0] * walls[i].coeficients[0] + vectorRay[1] * walls[i].coeficients[1] + vectorRay[2] * walls[i].coeficients[2];

        if (insideSqrt == 0) {
            continue;
        }

        distanceSol = -(point[0] * walls[i].coeficients[0] + point[1] * walls[i].coeficients[1] + point[2] * walls[i].coeficients[2] + walls[i].d) / insideSqrt;

        for (int j = 0; j < 3; j++) {
            newFirstPoint[j] = point[j] + vectorRay[j] * distanceSol;
            vector[j] = newFirstPoint[j] - point[j];

            if (vector[j]/vectorRay[j] < 0) {
                behind = 1;
                break;
            }
        }

        if (behind) {
            continue;
        }

        distanceSol = calculateDistance(point, newFirstPoint);

        if (distanceSol < distance) {
            distance = distanceSol;
            sphere = NULL;
            plane = walls + i;
            returnPoint[0] = newFirstPoint[0];
            returnPoint[1] = newFirstPoint[1];
            returnPoint[2] = newFirstPoint[2];
        }
    }

    if ((light.center[0] - point[0])/vectorRay[0] <= (light.center[1] - point[1])/vectorRay[1] + 0.0005 && 
        (light.center[0] - point[0])/vectorRay[0] >= (light.center[1] - point[1])/vectorRay[1] - 0.0005 && 
        (light.center[0] - point[0])/vectorRay[0] <= (light.center[2] - point[2])/vectorRay[2] + 0.0005 && 
        (light.center[0] - point[0])/vectorRay[0] >= (light.center[2] - point[2])/vectorRay[2] - 0.0005 && 
        (light.center[2] - point[2])/vectorRay[2] <= (light.center[1] - point[1])/vectorRay[1] + 0.0005 && 
        (light.center[2] - point[2])/vectorRay[2] >= (light.center[1] - point[1])/vectorRay[1] - 0.0005 && 
        calculateDistance(point, light.center) < distance) {

        sphere = NULL;
        plane = NULL;
        *lightRet = 1;
    }

    if (sphere != NULL) {
        for (int i = 0; i < 3; i++) {
            point[i] = returnPoint[i];
        }
    }
    else if (plane != NULL) {
        for (int i = 0; i < 3; i++) {
            point[i] = returnPoint[i];
        }
    }

    //PENSAR EN HACERLO RECURSIVO (ES MAS FACIL)

    *originSph = sphere;
    *originWall = plane;
}

void calculateReflex(ln *ray, double vector[3], double point[3]) {
    double newPoint[3];
    double std = sqrt(scalarProduct(vector, vector));
    /*For calculating the reflexed ray it is necesary to calculate a second point of the line.
    For calculating this point, we calculate a point of symmetry (a point in the line that goes 
    from the center of the sphere and the intersection point of the ray and the sphere).
    Then we calculate another point of the ray, so we can calculate the symmetric point of this 
    one using the point of symmetry.
    The last step is calling calculateRay passing this 2 points (the calculated and the 
    intersection point)*/
    for (int i = 0; i < 3; i++) {
        newPoint[i] = 2*(point[i] + vector[i]/std) - (point[i] - ray->vector[i]);
    }
    calculateRay(point, newPoint, ray);
}

void calculateRay(double point1[3], double point2[3], ln *ray) {
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

void writeImage() {
    fprintf(image, "P6 %d %d 255\n", width, height);
    fwrite(screen, 1, width*height*3, image);
}
