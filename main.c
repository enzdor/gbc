/*

         Trying to build gradient boosting with
         MSE as loss function and tree stump as
         weak learner.

*/

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define TRAIN_TIME 2
#define DEPTH 5 // number of stumps

typedef struct {
    float* values;
    size_t occupied;
    size_t capacity;
} dynamic_array;

void init_array(dynamic_array* arr, size_t initial_size)
{
    arr->values = malloc(initial_size * sizeof(float));
    arr->occupied = 0;
    arr->capacity = initial_size;
}

void reset_array(dynamic_array* arr)
{
    free(arr->values);
    init_array(arr, 0);
}

void add_element(dynamic_array* arr, float element)
{
    if (arr->capacity == 0) {
        arr->capacity = 1;
    } else if (arr->occupied >= arr->capacity) {
        arr->capacity *= 2;
        arr->values = realloc(arr->values, arr->capacity * sizeof(*arr->values));
    }

    arr->values[arr->occupied++] = element;
}

struct data {
    dynamic_array x;
    dynamic_array y;
    dynamic_array old_y;
    dynamic_array og_y;
    dynamic_array residuals;
};

struct node {
    dynamic_array xs;
    dynamic_array residuals;
    float mean;
    float output_value;
};

typedef struct {
    struct node* nodes;
    size_t occupied;
    size_t capacity;
} nodes_in_depth;

void init_depth_nodes(nodes_in_depth* nodes, size_t initial_size)
{
    nodes->nodes = malloc(initial_size * sizeof(struct node));
    nodes->occupied = 0;
    nodes->capacity = initial_size;
}

void add_node(nodes_in_depth* arr, struct node element)
{
    if (arr->capacity == 0) {
        arr->capacity = 1;
    } else if (arr->occupied >= arr->capacity) {
        arr->capacity *= 2;
        arr->nodes = realloc(arr->nodes, arr->capacity * sizeof(*arr->nodes));
    }

    arr->nodes[arr->occupied++] = element;
}

int get_n_columns(char* line)
{
    int n_cols = 0;
    char* tok;

    for (tok = strtok(line, ","); tok && *tok; tok = strtok(NULL, ",\n")) {
        n_cols++;
    }

    return n_cols;
}

int get_n_rows(FILE* fp)
{
    char* line = NULL;
    size_t line_len;
    int rows = 1;
    size_t read = 0;

    while (getline(&line, &line_len, fp) != -1) {
        rows++;
    }

    free(line);
    return rows;
}

const char* get_csv_element(char* line, int num)
{
    const char* tok;
    for (tok = strtok(line, ","); tok && *tok; tok = strtok(NULL, ",\n")) {
        if (--num == 0) {
            return tok;
        }
    }

    return NULL;
}

float get_mean(float* vals, int len)
{
    float sum = 0;

    for (int i = 0; i < len; i++) {
        sum += vals[i];
    }

    return sum / len;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s file.csv\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    char* home = getenv("HOME");
    FILE* fp;

    if (access(argv[1], F_OK) != 0) {
        fprintf(stderr, "error in access: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

    fp = fopen(argv[1], "r");
    if (fp == NULL) {
        fprintf(stderr, "error in fopen: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

    char* line = NULL;
    size_t line_len;

    int err = getline(&line, &line_len, fp);
    if (err == -1) {
        if (line) {
            free(line);
        }
        fclose(fp);
        fprintf(stderr, "error in getline: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

    // first col is y second is x
    int cols = get_n_columns(line);
    // number of observations
    int rows = get_n_rows(fp);

    float y_sum = 0, x_sum = 0, y_mean = 0, learning_rate = 0.1;

    struct data d;

    init_array(&d.x, rows);
    init_array(&d.y, rows);
    init_array(&d.og_y, rows);
    init_array(&d.old_y, rows);

    size_t read = 0;
    free(line);
    line = NULL;
    line_len = 0;

    rewind(fp);

    /*
     *
     *
     *
     * read stuff from csv and save into x and y
     *
     *
     *
     */

    for (; (read = getline(&line, &line_len, fp)) != -1;) {
        for (int j = 1; j <= 2; j++) {
            char* tmp = strdup(line);
            if (tmp == NULL) {
                free(line);
                fclose(fp);
                fprintf(stderr, "error in strdup: %s\n", strerror(errno));
                exit(EXIT_FAILURE);
            }

            float el = atof(get_csv_element(tmp, j));

            if (j == 1) {
                add_element(&d.y, el);
                add_element(&d.old_y, el);
                add_element(&d.og_y, el);
            }
            if (j == 2) {
                add_element(&d.x, el);
            }
            free(tmp);
        }
    }

    free(line);
    fclose(fp);

    if (errno != 0) {
        fprintf(stderr, "error in getline: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

    /*
     *
     *
     *
     *
     * build the tree
     *
     *
     *
     *
     *
     *
     */

    int max_nodes = round(pow(2, DEPTH));
    nodes_in_depth tree[DEPTH];

    for (int i = 0; i < DEPTH; i++) {
        int l = round(pow(2, i));
        init_depth_nodes(&tree[i], l);

        for (int j = 0; j < l; j++) {
            struct node empty;
            add_node(&tree[i], empty);
            init_array(&tree[i].nodes[j].xs, 0);
            init_array(&tree[i].nodes[j].residuals, 0);
            tree[i].nodes[j].mean = 0;
            tree[i].nodes[j].output_value = 0;
        }
    }

    for (int i = 0; i < d.x.occupied; i++) {
        add_element(&tree[0].nodes[0].xs, d.x.values[i]);
    }

    for (int depth = 1; depth < DEPTH; depth++) {
        int l = round(pow(2, depth));

        // for the number of nodes in x depth
        for (int curr_node_pos = 0; curr_node_pos < tree[depth - 1].occupied; curr_node_pos++) {
            // find mean?
            tree[depth - 1].nodes[curr_node_pos].mean = get_mean(tree[depth - 1].nodes[curr_node_pos].xs.values,
                tree[depth - 1].nodes[curr_node_pos].xs.occupied);

            // and split elements for the next depth
            for (int i = 0; i < tree[depth - 1].nodes[curr_node_pos].xs.occupied; i++) {
                int next_node_pos = (curr_node_pos + 1) * 2 - 1;

                if (tree[depth - 1].nodes[curr_node_pos].xs.values[i] < tree[depth - 1].nodes[curr_node_pos].mean) {
                    // left branch
                    add_element(&tree[depth].nodes[next_node_pos - 1].xs, tree[depth - 1].nodes[curr_node_pos].xs.values[i]);
                } else {
                    // right branch
                    add_element(&tree[depth].nodes[next_node_pos].xs, tree[depth - 1].nodes[curr_node_pos].xs.values[i]);
                }
            }
        }
    }

    /*
     *
     *
     *
     *
     * free x and residuals from tree?
     * need to free something for sure
     * trim unused nodes
     *
     *
     *
     *
     *
     */

    /*
     *
     *
     *
     *
     *
     *
     *
     *
     *
     *
     * do boosting
     *
     *
     *
     */

    time_t secs = TRAIN_TIME;
    time_t start_time = time(NULL);

    // a. init y as y mean

    y_mean = get_mean(d.y.values, d.y.occupied);

    for (int i = 0; i < d.y.occupied; i++) {
        d.y.values[i] = y_mean;
    }

    while (time(NULL) - start_time < secs) {
        // 1. compute pseudo residuals
        for (int i = 0; i < rows; i++) {
            add_element(&d.residuals, d.old_y.values[i] - d.y.values[i]);
            add_element(&tree[0].nodes[0].residuals, d.old_y.values[i] - d.y.values[i]);
        }

        // 2. fit regression tree to the pseudo residuals
        // top to bottom in a big ass for loop

        // need to check if n of residuals, ys and xs are the same
        for (int i = 0; i < d.x.occupied; i++) {
            int curr_node = 0;
            for (int depth = 1; depth < DEPTH; depth++) {
                curr_node = (curr_node + 1) * 2 - 2;
                if (d.x.values[i] > tree[depth - 1].nodes[curr_node].mean) {
                    curr_node++;
                }

                // add res of x to residuals in the current node
                add_element(&tree[depth].nodes[curr_node].residuals, d.residuals.values[i]);
            }
        }

        // 3. for each node, compute output value
        // (mean of residuals in node)

        for (int depth = 0; depth < DEPTH; depth++) {
            int l = round(pow(2, depth));
            for (int curr_node_pos = 0; curr_node_pos < l; curr_node_pos++) {
                if (tree[depth].nodes[curr_node_pos].residuals.occupied > 0) {
                    tree[depth].nodes[curr_node_pos].output_value = get_mean(
                        tree[depth].nodes[curr_node_pos].residuals.values,
                        tree[depth].nodes[curr_node_pos].residuals.occupied);
                }
            }
        }

        // 4. update y

        for (int i = 0; i < rows; i++) {
            int curr_node = 0;
            float adjustment = 0;

            for (int depth = 0; depth < DEPTH; depth++) {
                if (tree[depth].nodes[curr_node].residuals.occupied < 1) {
                    break;
                }

                adjustment = tree[depth].nodes[curr_node].output_value;

                curr_node = (curr_node + 1) * 2 - 2;
                if (d.x.values[i] > tree[depth].nodes[curr_node].mean) {
                    curr_node++;
                }
            }

            d.old_y.values[i] = d.y.values[i];
            d.y.values[i] = d.y.values[i] + learning_rate * adjustment;
        }
    }

    // for (int i = 0; i < inn.og_y.occupied; i++) {
    //     printf("real y %f\n", inn.og_y.values[i]);
    //     printf("predicted y %f and old y %f\n", inn.y.values[i], inn.old_y.values[i]);
    // }
}
