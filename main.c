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

#define TRAIN_TIME 300
#define DEPTH 5 // number of stumps

typedef struct {
    float* values;
    size_t occupied;
    size_t capacity;
} dynamic_array;

void init_array(dynamic_array* arr, size_t initial_size)
{
    if (initial_size < 1) {
        initial_size = 1;
    }

    arr->values = malloc(initial_size * sizeof(float));
    arr->occupied = 0;
    arr->capacity = initial_size;
}

void reset_array(dynamic_array* arr)
{
    free(arr->values);

    if (errno != 0) {
        fprintf(stderr, "error in free: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

    init_array(arr, 1);
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
    size_t n_nodes;
    size_t capacity;
} nodes_in_depth;

void init_depth_nodes(nodes_in_depth* nodes, size_t initial_size)
{
    nodes->nodes = malloc(initial_size * sizeof(struct node));
    nodes->n_nodes = 0;
    nodes->capacity = initial_size;
}

void add_node(nodes_in_depth* arr, struct node element)
{
    if (arr->capacity == 0) {
        arr->capacity = 1;
    } else if (arr->n_nodes >= arr->capacity) {
        arr->capacity *= 2;
        arr->nodes = realloc(arr->nodes, arr->capacity * sizeof(*arr->nodes));
    }

    arr->nodes[arr->n_nodes++] = element;
}

void free_tree(nodes_in_depth* tree, int depth_count)
{
    for (int i = 0; i < depth_count; i++) {
        int l = round(pow(2, i));
        for (int j = 0; j < l; j++) {
            free(tree[i].nodes[j].xs.values);
            free(tree[i].nodes[j].residuals.values);
        }
        free(tree[i].nodes);
    }
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
	if (len < 1) return 0;
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
    init_array(&d.residuals, 1);

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

    nodes_in_depth tree[DEPTH];
    struct node empty = { 0 };

    for (int i = 0; i < DEPTH; i++) {
        int l = round(pow(2, i));
        init_depth_nodes(&tree[i], l);

        for (int j = 0; j < l; j++) {
            add_node(&tree[i], empty);
            init_array(&tree[i].nodes[j].xs, 1);
            init_array(&tree[i].nodes[j].residuals, 1);
            tree[i].nodes[j].mean = 0;
            tree[i].nodes[j].output_value = 0;
        }
    }

    for (int i = 0; i < d.x.occupied; i++) {
        add_element(&tree[0].nodes[0].xs, d.x.values[i]);
    }

    for (int depth = 1; depth < DEPTH; depth++) {
        // for the number of nodes in x depth
        for (int curr_node_pos = 0; curr_node_pos < tree[depth - 1].n_nodes; curr_node_pos++) {
            // find mean
            tree[depth - 1].nodes[curr_node_pos].mean = get_mean(tree[depth - 1].nodes[curr_node_pos].xs.values,
                tree[depth - 1].nodes[curr_node_pos].xs.occupied);

            // and split elements for the next depth
            for (int i = 0; i < tree[depth - 1].nodes[curr_node_pos].xs.occupied; i++) {
                int left_child = curr_node_pos * 2;
                int right_child = curr_node_pos * 2 + 1;

                // make sure number of nodes is not exceeded
                if (left_child >= tree[depth].capacity || right_child >= tree[depth].capacity) {
                    continue;
                }

                if (tree[depth - 1].nodes[curr_node_pos].xs.values[i] < tree[depth - 1].nodes[curr_node_pos].mean) {
                    // left branch
                    add_element(&tree[depth].nodes[left_child].xs, tree[depth - 1].nodes[curr_node_pos].xs.values[i]);
                } else {
                    // right branch
                    add_element(&tree[depth].nodes[right_child].xs, tree[depth - 1].nodes[curr_node_pos].xs.values[i]);
                }
            }
        }
    }

    /*
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
        reset_array(&d.residuals);

        for (int i = 0; i < DEPTH; i++) {
            int l = round(pow(2, i));
            for (int j = 0; j < l; j++) {
                reset_array(&tree[i].nodes[j].residuals);
            }
        }

        // 1. compute pseudo residuals
        for (int i = 0; i < rows; i++) {
            add_element(&d.residuals, d.old_y.values[i] - d.y.values[i]);
            add_element(&tree[0].nodes[0].residuals, d.old_y.values[i] - d.y.values[i]);
        }

        if (!(d.y.occupied == d.residuals.occupied)) {
            fprintf(stderr, "error in boosting: expected number of xs, ys, and residuals to be equal, got %lu %lu %lu\n",
                d.residuals.occupied, d.y.occupied, d.x.occupied);
            exit(EXIT_FAILURE);
        }

        // 2. fit regression tree to the pseudo residuals
        for (int i = 0; i < d.x.occupied; i++) {
            int curr_node = 0;
            for (int depth = 1; depth < DEPTH; depth++) {
                curr_node = (curr_node + 1) * 2 - 2;
				// check if node exists and has data
                if (curr_node >= tree[depth - 1].n_nodes || tree[depth - 1].nodes[curr_node].xs.occupied < 1) {
                    break;
                }

                // calculate child indices
                int left_child = curr_node * 2;
                int right_child = curr_node * 2 + 1;

                // make sure children exist
                if (left_child >= tree[depth].capacity || right_child >= tree[depth].capacity) {
                    break;
                }

                if (d.x.values[i] <= tree[depth - 1].nodes[curr_node].mean) {
                    curr_node = left_child;
                } else {
                    curr_node = right_child;
                }

                // add residual to the current node
                if (curr_node < tree[depth].n_nodes) {
                    add_element(&tree[depth].nodes[curr_node].residuals, d.residuals.values[i]);
                }
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
            float adjustment = tree[0].nodes[0].output_value;

            for (int depth = 1; depth < DEPTH; depth++) {
                // Check bounds before accessing
                if (curr_node >= tree[depth - 1].n_nodes || tree[depth - 1].nodes[curr_node].xs.occupied < 1) {
                    break;
                }

                // Calculate child indices
                int left_child = curr_node * 2;
                int right_child = curr_node * 2 + 1;

                // Check child bounds
                if (left_child >= tree[depth].capacity || right_child >= tree[depth].capacity) {
                    break;
                }

                if (d.x.values[i] <= tree[depth - 1].nodes[curr_node].mean) {
                    curr_node = left_child;
                } else {
                    curr_node = right_child;
                }

                // use adjustment from this node if it has residuals
                if (curr_node < tree[depth].n_nodes && tree[depth].nodes[curr_node].residuals.occupied > 0) {
                    adjustment = tree[depth].nodes[curr_node].output_value;
                }
				printf("%f \n", adjustment);
            }

            d.old_y.values[i] = d.y.values[i];
            d.y.values[i] = d.y.values[i] + learning_rate * adjustment;
        }
    }

    for (int i = 0; i < d.og_y.occupied; i++) {
        printf("predicted y %f and old y %f\n", d.y.values[i], d.og_y.values[i]);
    }

    free(d.x.values);
    free(d.y.values);
    free(d.og_y.values);
    free(d.old_y.values);
    free(d.residuals.values);

    free_tree(tree, DEPTH);

    return 0;


}
