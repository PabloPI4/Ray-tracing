prog: ray-tracing.c
	gcc ray-tracing.c -lm -o ray-tracing
clean:
	rm -f ray-tracing image.ppm
