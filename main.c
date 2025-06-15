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

struct data {
    float* y;
    float* old_y;
    float* og_y;
    float* x;
    float* residuals;
    float* residuals_predictions;
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
    // x_l_mean, x_h_mean,
    learning_rate = 0.8;
    // columns are x and y

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

    // x y residual predicted_residuals

    // set mean and y as mean
    for (int i = 0; i < rows; i++) {
        old_y[i] = y[i];
        y[i] = y_mean;
        residuals[i] = old_y[i] - y[i];
    }

    time_t secs = TRAIN_TIME;
    time_t start_time = time(NULL);

    while (time(NULL) - start_time < secs) {
        // calculate residuals
        float left_sum = 0, right_sum = 0;

        for (int i = 0; i < rows; i++) {
            if (x[i] < x_mean) {
                left_sum += residuals[i];
            } else {
                right_sum += residuals[i];
            }
        }

        // float left_pred = (left_count > 0) ? left_sum / left_count : 0;
        // float right_pred = (right_count > 0) ? right_sum / right_count : 0;

        int max_leaves = round(pow(2, DEPTH));
        float* save_leaves[DEPTH][max_leaves];
        float save_means[DEPTH][max_leaves];
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

        for (int depth = 1; depth < DEPTH; depth++) {
            int l = round(pow(2, depth));

            // for the number of leafs in x depth
            for (int n_leaves = 0; n_leaves < l; n_leaves++) {
                // find mean
                float leaf_mean = 0;

				leaf_mean = get_mean(save_leaves[depth - 1][n_leaves], save_n[depth - 1][n_leaves]);

                // and split elements for the next depth
                for (int i = 0; i < save_n[depth - 1][n_leaves]; i++) {
                    int leaf_pos = (n_leaves + 1) * 2 - 1;
                    int l_count = 0;
                    int r_count = 0;

                    if (save_leaves[depth - 1][n_leaves][i] < leaf_mean) {
                        // left branch
                        save_leaves[depth][leaf_pos - 1][l_count] = save_leaves[depth - 1][n_leaves][i];

                        l_count++;
                        save_n[depth][leaf_pos - 1]++;
                    } else {
                        // right branch
                        save_leaves[depth][leaf_pos][r_count] = save_leaves[depth - 1][n_leaves][i];

                        r_count++;
                        save_n[depth][leaf_pos]++;
                    }
                }
            }
        }

        //
        //      float save_pseudo_residuals[inn.x.occupied];
        //
        //      // for each x
        //      for (int i = 0; i < inn.x.occupied; i++) {
        //          int end, current_leaf = 0;
        //          float current_mean = 0;
        //
        //          for (int depth = 0; depth < DEPTH; depth++) {
        //              int right_next_depth = current_leaf * 2 + 1;
        // 	printf("occupied %lu\n", save_leaves[depth][current_leaf].occupied);
        //              // check if there is a mean in this leaf
        //              if (save_leaves[depth][current_leaf].occupied < 1) {
        //                  break;
        //              }
        //
        //              current_mean = save_means[depth][current_leaf];
        //
        //              // save the pseudo residuals
        //              if (inn.x.values[i] < save_means[depth][current_leaf]) {
        //                  current_leaf = right_next_depth - 1;
        //              } else {
        //                  current_leaf = right_next_depth;
        //              }
        //          }
        //
        //          save_pseudo_residuals[i] = current_mean;
        // printf("%f save pseudo-residual, mean %f , current mean %f\n", save_pseudo_residuals[0], save_means[0][0], current_mean);
        // return 0;
        //      }

        /*
         *
         *
         *  CALCULATE MEANS FOR EACH LEAF
         *  pseudo residual
         *
         *
         */

        /*
         *
         *
         * 	FOR EACH Y GO DOWN TREE ACCORDING TO THEIR X
         * 	AND CALCULATE NEW Y ACCORDING TO THE MEAN OF THE
         * 	LEAF AND THE LEARNING RATE
         *
         * 	Y_SUB_i+1 = Y_SUB_i + learning rate * pseudo residual
         *
         *
         */

        // once i have the mean for all of the leafs
        // 		set the new y to the corresponding leaf by going down the tree
        // 		calculate residuals mean
        // 		set new y

        // for (int depth = 1; depth <= DEPTH; depth++) {
        //     Darray leaves[2];
        //     Darray next_leaves[(depth + 1) * 2];
        //     if (depth == 1) {
        //     }
        //     for (int i = 0; i < inn.og_y.occupied; i++) {
        //         float left_sum = 0, right_sum = 0;
        //
        //         for (int i = 0; i < inn.og_y.occupied; i++) {
        //             if (inn.x.values[i] < x_mean) {
        //                 left_sum += inn.residuals.values[i];
        //             } else {
        //                 right_sum += inn.residuals.values[i];
        //             }
        //         }
        //
        //         float left_pred = (left_count > 0) ? left_sum / left_count : 0;
        //         float right_pred = (right_count > 0) ? right_sum / right_count : 0;
        //
        //         Darray left_leaf, right_leaf;
        //
        //         init_array(&left_leaf, 0);
        //         init_array(&right_leaf, 0);
        //     }
        // }

        // for (int i = 0; i < inn.og_y.occupied; i++) {
        //     if (inn.x.values[i] < x_mean) {
        //         inn.residuals_predictions.values[i] = left_pred;
        //     } else {
        //         inn.residuals_predictions.values[i] = right_pred;
        //     }
        //
        //     inn.old_y.values[i] = inn.y.values[i];
        //     inn.y.values[i] = inn.y.values[i] + inn.residuals_predictions.values[i] * learning_rate;
        //     inn.residuals.values[i] = inn.old_y.values[i] - inn.y.values[i];
        // }
    }

    // for (int i = 0; i < inn.og_y.occupied; i++) {
    //     printf("real y %f\n", inn.og_y.values[i]);
    //     printf("predicted y %f and old y %f\n", inn.y.values[i], inn.old_y.values[i]);
    // }
}
