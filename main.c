/*

         Trying to build gradient boosting with
         MSE as loss function and tree stump as
         weak learner.

*/

#include <ctype.h>
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
} Darray;

struct input_data {
    Darray y;
    Darray old_y;
    Darray og_y;
    Darray x;
    Darray residuals;
    Darray residuals_predictions;
};

void init_array(Darray* arr, size_t initial_size)
{
    if (initial_size < 1) {
        initial_size = 1;
    }

    arr->values = malloc(initial_size * sizeof(float));
    arr->occupied = 0;
    arr->capacity = initial_size;
}

void add_element(Darray* arr, float element)
{
    if (arr->occupied >= arr->capacity) {
        arr->capacity *= 2;
        arr->values = realloc(arr->values, arr->capacity * sizeof(*arr->values));
    }

    arr->values[arr->occupied++] = element;
}

void delete_element(Darray* arr, float element)
{
    for (int i = 0; i < arr->occupied; i++) {
        if (arr->values[i] == element) {
            for (int j = i; j < arr->occupied; j++) {
                arr->values[j] = arr->values[j + 1];
            }
            arr->occupied--;
            arr->capacity--;
            break;
        }
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
	char* line;
	size_t line_len;
	int rows = 1;

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

int main(int argc, char* argv[])
{
    if (argc < 2) {
        puts("not enough arguments");
        exit(EXIT_FAILURE);
    }

    char* home = getenv("HOME");
    FILE* fp;

    if (access(argv[1], F_OK) != 0) {
        puts("access");
        return 1;
    }

    fp = fopen(argv[1], "r");
    if (fp == NULL) {
        puts("fopen");
        return 1;
    }

    char* line = NULL;
    size_t line_len;

	int err = getline(&line, &line_len, fp);
	if (err == -1) {
		puts("getline");
		return 1;
	}

	// first col is y second is x
	int cols = get_n_columns(line);
	printf("%d cols\n", cols);

	// number of observations
	int rows = get_n_rows(fp);
	printf("%d rows\n", rows);

    float y_sum = 0, x_sum = 0, y_mean, x_mean, learning_rate;
	// x_l_mean, x_h_mean,
    learning_rate = 0.8;
    // columns are x and y

    struct input_data inn;

    init_array(&inn.x, 0);
    init_array(&inn.y, 0);
    init_array(&inn.og_y, 0);
    init_array(&inn.old_y, 0);
    init_array(&inn.residuals, 0);
    init_array(&inn.residuals_predictions, 0);

    while (getline(&line, &line_len, fp) != -1) {
        for (int i = 1; i <= 2; i++) {
            char* tmp = strdup(line);
            float el = atof(get_csv_element(tmp, i));

            if (i == 1) {
                add_element(&inn.y, el);
                add_element(&inn.og_y, el);
            }
            if (i == 2) {
                add_element(&inn.x, el);
            }
            free(tmp);
        }
    }

    for (int i = 0; i < inn.og_y.occupied; i++) {
        add_element(&inn.old_y, 0);
        add_element(&inn.residuals, 0);
        add_element(&inn.residuals_predictions, 0);
    }

    // calculate mean
    for (int i = 0; i < inn.og_y.occupied; i++) {
        x_sum += inn.x.values[i];
        y_sum += inn.y.values[i];
    }

    y_mean = y_sum / inn.og_y.occupied;
    x_mean = x_sum / inn.og_y.occupied;
    // x y residual predicted_residuals

    // set mean and y as mean
    for (int i = 0; i < inn.og_y.occupied; i++) {
        inn.old_y.values[i] = inn.y.values[i];
        inn.y.values[i] = y_mean;
        inn.residuals.values[i] = inn.old_y.values[i] - inn.y.values[i];
    }

    time_t secs = TRAIN_TIME;
    time_t start_time = time(NULL);

    while (time(NULL) - start_time < secs) {
        // calculate residuals
        float left_sum = 0, right_sum = 0;

        for (int i = 0; i < inn.og_y.occupied; i++) {
            if (inn.x.values[i] < x_mean) {
                left_sum += inn.residuals.values[i];
            } else {
                right_sum += inn.residuals.values[i];
            }
        }

        // float left_pred = (left_count > 0) ? left_sum / left_count : 0;
        // float right_pred = (right_count > 0) ? right_sum / right_count : 0;

        int max_leaves = round(pow(2, DEPTH));
        Darray save_leaves[DEPTH][max_leaves];
        float save_means[DEPTH][max_leaves];

        for (int depth = 0; depth < DEPTH; depth++) {
            for (int j = 0; j < round(pow(2, depth + 1)); j++) {
                init_array(&save_leaves[depth][j], 0);
            }
        }

        save_leaves[0][0] = inn.x;


        for (int depth = 1; depth < DEPTH; depth++) {
            int l = round(pow(2, depth));

            Darray leaves[l];

            for (int i = 0; i < l; i++) {
                init_array(&leaves[i], 0);
            }

            // for the number of leafs in x depth
            for (int n_leaves = 0; n_leaves < l; n_leaves++) {
                // find mean
                float leaf_sum = 0, leaf_mean = 0;

                for (int i = 0; i < save_leaves[depth - 1][n_leaves].occupied; i++) {
                    leaf_sum += save_leaves[depth - 1][n_leaves].values[i];
                }

                leaf_mean = leaf_sum / save_leaves[depth - 1][n_leaves].occupied;
                save_means[depth - 1][n_leaves] = leaf_mean;

                // and split elements for the next depth
                for (int i = 0; i < save_leaves[depth - 1][n_leaves].occupied; i++) {
                    int leaf_pos = (n_leaves + 1) * 2 - 1;

                    if (save_leaves[depth - 1][n_leaves].values[i] < leaf_mean) {
                        // left branch
                        add_element(&save_leaves[depth][leaf_pos - 1], save_leaves[depth - 1][n_leaves].values[i]);
                    } else {
                        // right branch
                        add_element(&save_leaves[depth][leaf_pos], save_leaves[depth - 1][n_leaves].values[i]);
                    }
                }
            }
        }

        float save_pseudo_residuals[inn.x.occupied];

        // for each x
        for (int i = 0; i < inn.x.occupied; i++) {
            int end, current_leaf = 0;
            float current_mean = 0;

            for (int depth = 0; depth < DEPTH; depth++) {
                int right_next_depth = current_leaf * 2 + 1;
				printf("occupied %lu\n", save_leaves[depth][current_leaf].occupied);
                // check if there is a mean in this leaf
                if (save_leaves[depth][current_leaf].occupied < 1) {
                    break;
                }

                current_mean = save_means[depth][current_leaf];

                // save the pseudo residuals
                if (inn.x.values[i] < save_means[depth][current_leaf]) {
                    current_leaf = right_next_depth - 1;
                } else {
                    current_leaf = right_next_depth;
                }
            }

            save_pseudo_residuals[i] = current_mean;
			printf("%f save pseudo-residual, mean %f , current mean %f\n", save_pseudo_residuals[0], save_means[0][0], current_mean);
			return 0;
        }

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

    for (int i = 0; i < inn.og_y.occupied; i++) {
        printf("real y %f\n", inn.og_y.values[i]);
        printf("predicted y %f and old y %f\n", inn.y.values[i], inn.old_y.values[i]);
    }
}
