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
     * build the tree
     *
     *
     */

    int max_leaves = round(pow(2, DEPTH));
    float* save_leaves[DEPTH][max_leaves];
    float save_means[DEPTH][max_leaves];
    float save_output_value[DEPTH][max_leaves];
    int save_n[DEPTH][max_leaves];

    for (int i = 0; i < DEPTH; i++) {
        for (int j = 0; j < max_leaves; j++) {
            save_n[i][j] = 0;
            save_means[i][j] = 0;
            save_leaves[i][j] = x;
        }
    }

    save_n[0][0] = rows;
    save_leaves[0][0] = x;
    save_means[0][0] = x_mean;

    for (int depth = 1; depth < DEPTH; depth++) {
        int l = round(pow(2, depth));

        // for the number of leafs in x depth
        for (int curr_leaf_pos = 0; curr_leaf_pos < l; curr_leaf_pos++) {
            // find mean
            float leaf_mean = 0;

            leaf_mean = get_mean(save_leaves[depth - 1][curr_leaf_pos], save_n[depth - 1][curr_leaf_pos]);
            save_means[depth - 1][curr_leaf_pos] = leaf_mean;

            // and split elements for the next depth
            for (int i = 0; i < save_n[depth - 1][curr_leaf_pos]; i++) {
                int next_leaf_pos = (curr_leaf_pos + 1) * 2 - 1;
                int l_count = 0;
                int r_count = 0;

                if (save_leaves[depth - 1][curr_leaf_pos][i] < leaf_mean) {
                    // left branch
                    save_leaves[depth][next_leaf_pos - 1][l_count] = save_leaves[depth - 1][curr_leaf_pos][i];

                    l_count++;
                    save_n[depth][next_leaf_pos - 1]++;
                } else {
                    // right branch
                    save_leaves[depth][next_leaf_pos][r_count] = save_leaves[depth - 1][curr_leaf_pos][i];

                    r_count++;
                    save_n[depth][next_leaf_pos]++;
                }
            }
        }
    }

    /*
     *
     *
     *
     * output value
     *
     *
     *
     */

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

    while (time(NULL) - start_time < secs) {
        // 1. compute pseudo residuals
        for (int i = 0; i < rows; i++) {
            residuals[i] = old_y[i] - y[i];
        }

        // 2. fit regression tree to the pseudo residuals
        // top to bottom in a big ass for loop
        // save into an array

        float* res_in_leaf[DEPTH][max_leaves];

        // initialize res_in_leaf;

        res_in_leaf[0][0] = residuals;

        for (int i = 0; i < rows; i++) {
            int curr_leaf = 0;
            for (int depth = 1; depth < DEPTH; depth++) {
                if (x[i] < save_means[depth][curr_leaf]) {
                    // add x to res_in_leaf

                    curr_leaf = (curr_leaf + 1) * 2 - 2;
                    if (save_n[depth][curr_leaf] < 1) {
                        break;
                    }
                } else {
                    curr_leaf++;
                    // add x to res_in_leaf

                    curr_leaf = (curr_leaf + 1) * 2 - 2;
                    if (save_n[depth][curr_leaf] < 1) {
                        break;
                    }
                }
            }
        }

        // 3. for each leaf, compute output value
        // (mean of residuals in leaf)

        float output_values[DEPTH][max_leaves];
        for (int i = 0; i < DEPTH; i++) {
            for (int j = 0; j < max_leaves; j++) {
                output_values[i][j] = 0;
            }
        }

        for (int depth = 0; depth < DEPTH; depth++) {
            int l = round(pow(2, depth));
            for (int curr_leaf_pos = 0; curr_leaf_pos < l; curr_leaf_pos++) {
                if (save_n[depth][curr_leaf_pos] > 0) {
                    // calculate mean
                }
            }
        }

        // 4. update y

        for (int i = 0; i < rows; i++) {
            int curr_leaf = 0;
            float adjustment = 0;

            for (int depth = 1; depth < DEPTH; depth++) {
                if (x[i] < save_means[depth][curr_leaf]) {
                    adjustment = output_values[depth][curr_leaf];
                    curr_leaf = (curr_leaf + 1) * 2 - 2;
                    if (save_n[depth][curr_leaf] < 1) {
                        break;
                    }

                } else {
                    curr_leaf++;
                    adjustment = output_values[depth][curr_leaf];
                    curr_leaf = (curr_leaf + 1) * 2 - 2;
                    if (save_n[depth][curr_leaf] < 1) {
                        break;
                    }
                }
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
