/**
 * Starting point code for Project #9 (Parallel Game o' Life)
 * Simulates game of life from parameters passed by a text file from
 * command line.
 *
 * This code is a part of COMP280 Project #9
 * 
 * Authors:
 * 	 - Sat Garcia (starter)
 * 	 - Chris Eardensohn(ceardensohn@sandiego.edu)
 * 	 - Carolina Canales(ccanalesvillarreal@sandiego.edu)
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>

// C doesn't have booleans so we resort to macros. BOO!
#define LIVE 1
#define DEAD 0

// You can modify these if you want your printed board to look a bit
// different.
#define LIVECHAR '@'
#define DEADCHAR '.'

// Sleep for 0.2 seconds (i.e. 200,000us) between turns
#define SLEEPTIME_US 200000

//forward declarations
void *threadDoTurn(void* arg);
void *threadSimulate(void* arg);
void *printThreadPartition(void* arg);	
void timeval_subtract (struct timeval *result, 
						struct timeval *x, struct timeval *y);

struct ThreadArgs {
	unsigned width;
	unsigned height;
	unsigned num_iters;
	unsigned start_row;
	unsigned end_row;
	unsigned rank;
	unsigned print_partition;
	int print_world;
	int* world;
	pthread_barrier_t* barrier;
};
typedef struct ThreadArgs ThreadArgs;



/**
 * Given 2D coordinates, compute the corresponding index in the 1D array.
 *
 * @param x The x-coord we are converting (i.e. column number)
 * @param y The y-coord we are converting (i.e. row number)
 * @param width The width of the world (i.e. number of columns)
 * @param height The height of the world (i.e. number of rows)
 * @return Index into the 1D world array that corresponds to (x,y)
 */
static unsigned computeIndex(int x, int y, unsigned width, unsigned height) {
	// If x or y coordinates are out of bounds, wrap them arounds
	if (x < 0) {
		x += width;
	}
	else if (x >= (int)width) {
		x -= width;
	}

	if (y < 0) {
		y += height;
	}
	else if (y >= (int)height) {
		y -= height;
	}

	return y*width + x;
}

/**
 * Prints out the world.
 *
 * @note The exact characters being printed come from the LIVECHAR and
 * DEADCHAR macros
 *
 * @param world The world to print.
 * @param width The width of world.
 * @param height The height of world.
 */
static void printWorld(int *world, unsigned width, unsigned height) {
	unsigned i, j;
	unsigned index = 0;
	for (i = 0; i < width; ++i) {
		for (j = 0; j < height; ++j, ++index) {
			if (world[index] == LIVE)
				fprintf(stdout, "%c", LIVECHAR);
			else
				fprintf(stdout, "%c", DEADCHAR);
		}
		fprintf(stdout, "\n");
	}
}

/**
 * Creates and initializes the world based with a given set of cells which are
 * alive at the beginning.
 *
 * @note The set of live cells is given as an index into the world array, not
 * as a set of (x,y) coordinates
 *
 * @param width The width of the world.
 * @param height The height of the world.
 * @param init_set Array of cells that are alive at the beginning
 * @param init_set_size Number of elements in the init_set array
 * @return A 1D board that represents the initialized world.
 */
int* initWorld(unsigned width, unsigned height, 
				unsigned *init_set, unsigned init_set_size) {
	int *world = calloc(width*height, sizeof(int));
	unsigned i;
	for (i = 0; i < init_set_size; ++i) {
		world[init_set[i]] = LIVE;
	}
	return world;
}

/**
 * Returns the number of neighbors around a given (x,y) point that are alive.
 *
 * @param world The world we are simulating
 * @param x x-coord whose neighbors we are examining
 * @param y y-coord whose neighbors we are examining
 * @param width The width of the world
 * @param height The height of the world
 * @return The number of live neighbors around (x,y)
 */
static unsigned getNumLiveNeighbors(int *world, int x, int y,
									unsigned width, unsigned height) {
	unsigned sum = 0;

	for (int i = x-1; i <= x+1; i++) {
		for (int j = y-1; j <= y+1; j++) {
			if (i == x && j == y) continue;
			unsigned index = computeIndex(i,j,width,height);
			if (world[index] == LIVE) {
				++sum;
			}
		}
	}

	return sum;
}


/**
 * Updates cell at given coordinate.
 *
 * @param curr_world World for the current turn (read-only).
 * @param next_world World for the next turn.
 * @param x x-coord whose neighbors we are examining
 * @param y y-coord whose neighbors we are examining
 * @param width The width of the world
 * @param height The height of the world
 */
static void computeCell(int *curr_world, int *next_world, int x, int y, 
							int width, int height) {
	
	unsigned index = computeIndex(x, y, width, height);
	unsigned num_live_neighbors = getNumLiveNeighbors(curr_world, x, y, width, height);
	if (curr_world[index] == LIVE
			&& (num_live_neighbors < 2 || num_live_neighbors > 3)) {
		/*
		 * With my cross-bow,
		 * I shot the albatross.
		 */
		next_world[index] = DEAD;
	}
	else if (num_live_neighbors == 3) {
		/*
		 * Oh! Dream of joy! Is this indeed
		 * The light-house top I see?
		 */
		next_world[index] = LIVE;
	}
}

/**
 * Prints a helpful message for how to use the program.
 *
 * @param prog_name The name of the program as called from the command line.
 */
static void usage(char *prog_name) {
	fprintf(stderr, "usage: %s [-v] -c <config-file> -t <num_threads> -p\n", prog_name);
	exit(1);
}

/**
 * Creates a world based on the given configuration file.
 *
 * @param config_filename The name of the file which contains the configuration
 * 	for the world, including its initial state.
 * @param width Location to store the world width that is read in from the file.
 * @param height Location to store the world height that is read in from the file.
 * @param num_iters Location to store the number of simulation iterations.
 * @return The newly created world.
 */
static int *createWorld(char *config_filename, unsigned *width, unsigned *height,
						unsigned *num_iters) {
	FILE *config_file = fopen(config_filename, "r");

	unsigned init_set_size;
	fscanf(config_file, "%u", width);
	fscanf(config_file, "%u", height);
	fscanf(config_file, "%u", num_iters);
	fscanf(config_file, "%u", &init_set_size);

	unsigned x, y;
	unsigned i = 0;
	unsigned *init_set = malloc(init_set_size*sizeof(unsigned));

	while (fscanf(config_file, "%u %u", &x, &y) != EOF) {
		if (i == init_set_size) {
			fprintf(stderr, "ran out of room for coords\n");
			exit(1);
		}

		init_set[i] = computeIndex(x, y, *width, *height);
		++i;
	}

	fclose(config_file);

	assert(i == init_set_size);

	int *world = initWorld(*width, *height, init_set, init_set_size);

	free(init_set);

	return world;
}

int main(int argc, char *argv[]) {
	int print_world = 0;
	int print_partition = 0;
	int num_workers = 4;
	char *filename = NULL;
	char ch;
	pthread_barrier_t barrier;

	while ((ch = getopt(argc, argv, "vc:t:p")) != -1) {
		switch (ch) {
			case 'v':
				print_world = 1;
				break;
			case 'c':
				filename = optarg;
				break;
			case 't':
				if(strtol(optarg, NULL, 10) <= 0) {
					printf("Error: thread count must be a positive integer.\n");
					exit(1);
				}
				num_workers = strtol(optarg, NULL, 10);
				break;
			case 'p':
				print_partition = 1;
				break;
			default:
				usage(argv[0]);
		}
	}
	
	// print usage information to console	
	if (filename == NULL) {
		usage(argv[0]);
	}

	// create world
	unsigned width, height, num_iters;
	int *world = createWorld(filename, &width, &height, &num_iters);
	
	// catch error for more threads than rows
	if(num_workers > (int)height) {
		printf("Error: Number of threads must be no more "
			   	"than the height (%d) of the game board.\n",
			   	height);
		exit(1);
	}	

	// start timing
	struct timeval start_time, end_time, elapsed_time;
	gettimeofday(&start_time, NULL);
	
	//initialize workers and their structs
	ThreadArgs *worker_struct = malloc(num_workers*sizeof(ThreadArgs));
	pthread_t *worker = malloc(num_workers*sizeof(pthread_t));

	
	// initialize barrier
	int ret = pthread_barrier_init(&barrier, NULL, num_workers);
	if (ret != 0){
		perror("pthread_barrier_init error\n");
		exit(1);
	}
	
	//initialize structs
	unsigned remainder = height % num_workers;
	unsigned current_row = 0;
	unsigned rows_per_thread = height/num_workers;
	int i = 0;
	for(i = 0; i < num_workers; i++) {
		worker_struct[i].width = width;
		worker_struct[i].height = height;
		worker_struct[i].barrier = &barrier;
		worker_struct[i].num_iters = num_iters;
		worker_struct[i].world = world;
		worker_struct[i].print_world = 0;
		worker_struct[i].print_partition = print_partition;
		worker_struct[i].rank = i;
		if (remainder > 0) {
			worker_struct[i].start_row = current_row;
			worker_struct[i].end_row = current_row + rows_per_thread;
			remainder--;
			current_row = worker_struct[i].end_row + 1;	
		}
		else {
			worker_struct[i].start_row = current_row;
			worker_struct[i].end_row = current_row + rows_per_thread - 1;
			current_row = worker_struct[i].end_row + 1;
		}
	}

	worker_struct[0].print_world = print_world;
	
	//send threads out into the world of life
	for(i=0; i < num_workers; i++) {
		pthread_create(&worker[i], NULL, threadSimulate, &worker_struct[i]);
	}
		
	// join threads
	for(i = 0; i < num_workers; i++) {
		pthread_join(worker[i], NULL);
	}

	if(pthread_barrier_destroy(&barrier) == 1){
		perror("pthread_barrier_destroy error\n");
		exit(1);
	}

	// end timing
	gettimeofday(&end_time, NULL);
	timeval_subtract(&elapsed_time, &end_time, &start_time);
	printf("Total time for %d iterations of %dx%d world is %ld.%06ld\n",
			num_iters, width, height, elapsed_time.tv_sec, (long)elapsed_time.tv_usec);
	
	free(worker_struct);	
	free(worker);
	free(world);

	return 0;
}

/**
 * Multi-thread version of simulate to run game of life. Simulates a world of
 * given parameters and nummber of turns passed by struct.
 * @param arg the ThreadArgs struct to contain data to pass into the world
 */
void *threadSimulate(void* arg){
	struct ThreadArgs *worker_struct = (struct ThreadArgs*) arg;
	unsigned i = 0;
	do {
		pthread_barrier_wait(worker_struct->barrier);
		if (worker_struct->print_world) {
			system("clear");
			fprintf(stdout, "Time step: %u\n", i);
			printWorld(worker_struct->world, worker_struct->width, worker_struct->height);
			usleep(SLEEPTIME_US);
		}	
		threadDoTurn(arg);	
		i++;
	} while ( i <= worker_struct->num_iters );
	printThreadPartition(arg);
	return NULL;
}

/**
 * Prints each thread's logical id and the rows it was allocated
 * @param arg the worker struct containing the thread info including thread
 * rank, start row and end row
 */
void *printThreadPartition(void* arg){	
	struct ThreadArgs *worker_struct = (struct ThreadArgs*) arg;
	fprintf(stdout, "tid %d, rows: %d -> %d (%d)\n", worker_struct->rank,
			worker_struct->start_row, worker_struct->end_row,
			(worker_struct->end_row - worker_struct->start_row + 1));
	fflush(stdout);
	return NULL;
}

/**
 * Perform turns for each thread. Uses thread's world and start and end row  to
 * split the load
 * @arg the struct containing each thread's info and world board
 */
void *threadDoTurn(void* arg) {
	struct ThreadArgs *t_struct = (struct ThreadArgs*) arg;
	int *world_copy = malloc(t_struct->width*t_struct->height*sizeof(int));	
	for (unsigned i = 0; i < t_struct->width*t_struct->height; i++) {	
		world_copy[i] = t_struct->world[i];
	}
	// hold for threads
	pthread_barrier_wait(t_struct->barrier);
	for(int j = t_struct->start_row; j <= (int)t_struct->end_row; j++) {
		for(int i = 0; i < (int)t_struct->width; i++) {
			computeCell(world_copy, t_struct->world, i, j, t_struct->width, t_struct->height);
		}
	}
	free(world_copy);
	return NULL;
}

/**
 * Subtracts two timevals, storing the result in third timeval.
 *
 * @param result The result of the subtraction.
 * @param end The end time (i.e. what we are subtracting from)
 * @param start The start time (i.e. the one being subtracted)
 *
 * @url https://www.gnu.org/software/libc/manual/html_node/Elapsed-Time.html
 */
void timeval_subtract (struct timeval *result, 
						struct timeval *end, struct timeval *start)
{
	// Perform the carry for the later subtraction by updating start.
	if (end->tv_usec < start->tv_usec) {
		int nsec = (start->tv_usec - end->tv_usec) / 1000000 + 1;
		start->tv_usec -= 1000000 * nsec;
		start->tv_sec += nsec;
	}
	if (end->tv_usec - start->tv_usec > 1000000) {
		int nsec = (end->tv_usec - start->tv_usec) / 1000000;
		start->tv_usec += 1000000 * nsec;
		start->tv_sec -= nsec;
	}

	// Compute the time remaining to wait.tv_usec is certainly positive.
	result->tv_sec = end->tv_sec - start->tv_sec;
	result->tv_usec = end->tv_usec - start->tv_usec;
}
