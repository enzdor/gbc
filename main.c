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

struct node {
    float* values;
    float* residuals;
    int size_v;
    int size_r;
    float mean;
    float output_value;
};

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

    float y_sum = 0, x_sum = 0, y_mean, x_mean, learning_rate;
    learning_rate = 0.8;

    float x[rows];
    float y[rows];
    float og_y[rows];
    float old_y[rows];
    float residuals[rows];
    float residuals_predictions[rows];

    for (int i = 0; i < rows; i++) {
        x[i] = 0;
        y[i] = 0;
        og_y[i] = 0;
        old_y[i] = 0;
        residuals[i] = 0;
        residuals_predictions[i] = 0;
    }

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

    for (int i = 0; (read = getline(&line, &line_len, fp)) != -1; i++) {
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
                y[i] = el;
                og_y[i] = el;
                old_y[i] = el;
            }
            if (j == 2) {
                x[i] = el;
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

    // calculate mean
    y_mean = get_mean(y, rows);
    x_mean = get_mean(x, rows);

    // set mean and y as mean
    for (int i = 0; i < rows; i++) {
        y[i] = y_mean;
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
    struct node* tree[DEPTH];

    struct node nodes_0[1] = {
        { x, 0, rows, rows, 0, 0 }
    };
    struct node node_e = { 0, 0, 0, 0, 0, 0 };

    nodes_0[0].mean = get_mean(nodes_0[0].values, nodes_0[0].size_v);

    tree[0] = malloc(sizeof(nodes_0));
    tree[0] = nodes_0;

    for (int depth = 1; depth < DEPTH; depth++) {
        int l = round(pow(2, depth));
        tree[depth] = malloc(l * sizeof(node_e));

        for (int curr_node_pos = 0; curr_node_pos < l; curr_node_pos++) {

            tree[depth][curr_node_pos] = node_e;
            tree[depth][curr_node_pos].values = malloc(rows * sizeof(float));
            tree[depth][curr_node_pos].values[0] = 0;
            tree[depth][curr_node_pos].residuals = malloc(rows * sizeof(float));
            tree[depth][curr_node_pos].residuals[0] = 0;
        }
    }

    for (int depth = 1; depth < DEPTH; depth++) {
        int l = round(pow(2, depth));

        // for the number of nodes in x depth
        for (int curr_node_pos = 0; curr_node_pos < l; curr_node_pos++) {
            int l_count = 0;
            int r_count = 0;
			printf("depth %d, node %d\n", depth, curr_node_pos);

            // and split elements for the next depth
            for (int i = 0; i < tree[depth - 1][curr_node_pos].size_v; i++) {
                int next_node_pos = (curr_node_pos + 1) * 2 - 1;

                if (tree[depth - 1][curr_node_pos].values[i] < tree[depth - 1][curr_node_pos].mean) {
                    // left branch
                    tree[depth][next_node_pos - 1].values[l_count] = tree[depth - 1][curr_node_pos].values[i];
                    tree[depth][next_node_pos - 1].size_v++;
                    l_count++;
                } else {
                    // right branch
                    tree[depth][next_node_pos].values[r_count] = tree[depth - 1][curr_node_pos].values[i];
                    tree[depth][next_node_pos].size_v++;
                    r_count++;
                }
            }
        }

        for (int curr_node_pos = 0; curr_node_pos < l; curr_node_pos++) {
			for (int i = 1; i > -1; i--) {
                int next_node_pos = (curr_node_pos + 1) * 2 - 1;

				tree[depth][next_node_pos - i].mean = get_mean(tree[depth][next_node_pos - i].values, 
						tree[depth][next_node_pos].size_v);
			}
		}
    }

    /*
     *
     *
     *
     *
     *
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

    while (time(NULL) - start_time < secs) {
        // 1. compute pseudo residuals
        for (int i = 0; i < rows; i++) {
            residuals[i] = old_y[i] - y[i];
        }

        // 2. fit regression tree to the pseudo residuals
        // top to bottom in a big ass for loop
        // save into an array

        // initialize res_in_node;

        tree[0][0].residuals = residuals;

        for (int i = 0; i < rows; i++) {
            int curr_node = 0;
            for (int depth = 1; depth < DEPTH; depth++) {
                if (x[i] > tree[depth][curr_node].mean) {
                    curr_node++;
                }

                // add res of x to residuals in the current node
                tree[depth][curr_node].values[tree[depth][curr_node].size_r++] = residuals[i];
                curr_node = (curr_node + 1) * 2 - 2;
            }
        }

        // 3. for each node, compute output value
        // (mean of residuals in node)

        for (int depth = 0; depth < DEPTH; depth++) {
            int l = round(pow(2, depth));
            for (int curr_node_pos = 0; curr_node_pos < l; curr_node_pos++) {
                if (tree[depth][curr_node_pos].size_r > 0) {
                    tree[depth][curr_node_pos].mean = get_mean(tree[depth][curr_node_pos].residuals,
                        tree[depth][curr_node_pos].size_r);
                }
            }
        }

        // 4. update y

        for (int i = 0; i < rows; i++) {
            int curr_node = 0;
            float adjustment = 0;

            for (int depth = 1; depth < DEPTH; depth++) {
                if (tree[depth][curr_node].size_r < 1) {
                    break;
                }
                if (x[i] < tree[depth][curr_node].mean) {
                    curr_node++;
                }

                adjustment = tree[depth][curr_node].output_value;
                curr_node = (curr_node + 1) * 2 - 2;
            }

            old_y[i] = y[i];
            y[i] = y[i] + learning_rate * adjustment;
        }
    }

    // for (int i = 0; i < inn.og_y.occupied; i++) {
    //     printf("real y %f\n", inn.og_y.values[i]);
    //     printf("predicted y %f and old y %f\n", inn.y.values[i], inn.old_y.values[i]);
    // }
}
