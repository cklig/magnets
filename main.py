import numpy as np
import random
import math


def offset_multiply(matrix1, matrix2, x_offset, y_offset):
    rows, cols = matrix1.shape
    x_end = cols - abs(x_offset)
    y_end = rows - abs(y_offset)

    if x_offset >= 0:
        x_start = x_offset
        x_end = cols
    else:
        x_start = 0
        x_end = cols + x_offset

    if y_offset >= 0:
        y_start = y_offset
        y_end = rows
    else:
        y_start = 0
        y_end = rows + y_offset

    overlapping_matrix1 = matrix1[y_start:y_end, x_start:x_end]
    overlapping_matrix2 = matrix2[-1 * min(y_offset, 0):-1 * min(y_offset, 0) + overlapping_matrix1.shape[0],
                                  -1 * min(x_offset, 0):-1 * min(x_offset, 0) + overlapping_matrix1.shape[1]]

    result = np.multiply(overlapping_matrix1, overlapping_matrix2)
    return result


def get_offset_force_matrix(matrix_a, matrix_b, width):
    offset_force_matrix = np.zeros((2 * width - 1, 2 * width - 1))

    for x_offset in range(1 - width, width):
        for y_offset in range(1 - width, width):
            row = y_offset - 1 + width
            col = x_offset - 1 + width
            offset_force_matrix[row][col] = sum(offset_multiply(matrix_a, matrix_b, x_offset, y_offset).flatten())

    return offset_force_matrix


# force_values[0] is force if aligned (- for attraction), force_values[1] is sorted list of all other force values
def get_force_values(matrix_a, matrix_b, width):
    aligned_index = int(((2 * width - 1) ** 2 - 1) / 2)
    force_values = np.concatenate((get_offset_force_matrix(matrix_a, matrix_b, width).flatten(),
                                  get_offset_force_matrix(np.rot90(matrix_a, k=1), matrix_b, width).flatten(),
                                  get_offset_force_matrix(np.rot90(matrix_a, k=2), matrix_b, width).flatten(),
                                  get_offset_force_matrix(np.rot90(matrix_a, k=3), matrix_b, width).flatten()))
    return force_values[aligned_index], np.sort(np.delete(force_values, aligned_index))


WIDTH = 20


current_best = 1000
attained_target = False
number_of_attempts = 0
number_of_changes = WIDTH ** 2
attempts_since_previous_best = 0
best_surfaces = []
force_improve_timer = 0

while not attained_target:

    number_of_attempts = number_of_attempts + 1
    attempts_since_previous_best = attempts_since_previous_best + 1

    if number_of_attempts < 1000:
        surface_a = np.random.randint(-1, 2, (WIDTH, WIDTH))
        #surface_b = np.random.randint(-1, 2, (WIDTH, WIDTH))
        surface_b = np.zeros((WIDTH, WIDTH)) - surface_a
    else:
        surface_a = current_best_surface_a
        surface_b = current_best_surface_b
        number_of_changes = np.random.randint(1, WIDTH ** 2)

        breed_with_index = np.random.randint(1, len(best_surfaces) - 1)
        breed_surface_a = np.array(best_surfaces[breed_with_index][0])
        breed_surface_b = np.array(best_surfaces[breed_with_index][1])
        for index in random.sample(range(0, WIDTH ** 2), number_of_changes):
            row = int(index % WIDTH)
            col = int((index - row) / WIDTH)
            surface_a[col][row] = breed_surface_a[col][row]
            surface_b[col][row] = breed_surface_b[col][row]
        if attempts_since_previous_best > 100 and attempts_since_previous_best % 2 == 0:
            for i in range(np.random.randint(1, 1 + int(round(math.sqrt(attempts_since_previous_best)/10)))):
                row = np.random.randint(0, WIDTH)
                col = np.random.randint(0, WIDTH)
                if surface_a[col][row] * surface_b[col][row] == 0:
                    if attempts_since_previous_best % 4 == 0:
                        surface_a[col][row] = 1
                        surface_b[col][row] = -1
                    else:
                        surface_a[col][row] = -1
                        surface_b[col][row] = 1
                if surface_a[col][row] * surface_b[col][row] == 1:
                    if attempts_since_previous_best % 4 == 0:
                        surface_a[col][row] = -1 * surface_a[col][row]
                    else:
                        surface_b[col][row] = -1 * surface_b[col][row]

    force_values = get_force_values(surface_a, surface_b, WIDTH)
    #attempt_value = force_values[0] / max(-1 * force_values[1][0], force_values[1][-1])
    attempt_value = force_values[0] / max(-1 * force_values[1][0], 0)

    if current_best > -5:
        force_improve_cutoff = 5
    else:
        if current_best > -10:
            force_improve_cutoff = 2
        else:
            force_improve_cutoff = -1

    if (attempt_value <= current_best and force_improve_timer < force_improve_cutoff) or attempt_value < current_best:
        if attempt_value < current_best:
            force_improve_timer = 0
        if attempt_value == current_best:
            force_improve_timer = force_improve_timer + 1
        print("New Optimal (Value = ", attempt_value,
              ", Attempt #", number_of_attempts,
              ", Number of Changes: ", number_of_changes, ")")
        print(surface_a.tolist())
        print(surface_b.tolist())
        current_best = attempt_value
        current_best_surface_a = surface_a
        current_best_surface_b = surface_b
        if len(best_surfaces) > 0:
            if not (np.array_equal(surface_a, np.array(best_surfaces[0][0])) and
                    np.array_equal(surface_b, np.array(best_surfaces[0][1]))):
                best_surfaces = [[surface_a.tolist(), surface_b.tolist()]] + best_surfaces
        else:
            best_surfaces = [[surface_a.tolist(), surface_b.tolist()]] + best_surfaces
        print(len(best_surfaces))
        if len(best_surfaces) > 20:
            best_surfaces = best_surfaces[0:20]
        attempts_since_previous_best = 0

    if current_best < -50:
        attained_target = True






