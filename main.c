/*

         Trying to build gradient boosting with
         MSE as loss function and tree stump as
         weak learner.

*/

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

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
    arr->values = malloc(initial_size * sizeof(float));
    arr->occupied = 0;
    arr->capacity = initial_size;
}

void add_element(Darray* arr, float element)
{
    if (arr->capacity == 0) {
        arr->capacity = 1;
    } else if (arr->occupied >= arr->capacity) {
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

    float y_sum = 0, x_sum = 0, y_mean, x_mean, x_l_mean, x_h_mean, learning_rate;
    learning_rate = 0.8;
    // columns are x and y

    struct input_data inn;

    init_array(&inn.x, 0);
    init_array(&inn.y, 0);
    init_array(&inn.og_y, 0);
    init_array(&inn.old_y, 0);
    init_array(&inn.residuals, 0);
    init_array(&inn.residuals_predictions, 0);

    char* line = NULL;
    size_t line_len;

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

    int left_count = 0, right_count = 0;
    for (int i = 0; i < inn.og_y.occupied; i++) {
        if (inn.x.values[i] < x_mean) {
            left_count++;
        } else {
            right_count++;
        }
    }

    time_t secs = 600 * 6;
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

        float left_pred = (left_count > 0) ? left_sum / left_count : 0;
        float right_pred = (right_count > 0) ? right_sum / right_count : 0;

        for (int i = 0; i < inn.og_y.occupied; i++) {
            if (inn.x.values[i] < x_mean) {
                inn.residuals_predictions.values[i] = left_pred;
            } else {
                inn.residuals_predictions.values[i] = right_pred;
            }

            inn.old_y.values[i] = inn.y.values[i];
            inn.y.values[i] = inn.y.values[i] + inn.residuals_predictions.values[i] * learning_rate;
            if (i == 1) {
		//               printf("y is %f and real y %f\n", inn.y.values[i], inn.og_y.values[i]);
		// printf("preds %f and %f\n", left_pred, right_pred);
            }
            inn.residuals.values[i] = inn.old_y.values[i] - inn.y.values[i];
        }
    }

    for (int i = 0; i < inn.og_y.occupied; i++) {
        printf("real y %f\n", inn.og_y.values[i]);
        printf("predicted y %f and old y %f\n", inn.y.values[i], inn.old_y.values[i]);
    }
    printf("%lu\n", inn.og_y.occupied);
}
