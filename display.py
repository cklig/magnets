import numpy as np
import matplotlib.pyplot as plt


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
surface_a_list = [[0, -1, -1, 0, 0, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 0, 0, 1], [0, 0, 0, 1, 1, 1, 1, 0, -1, -1, -1, 1, -1, 0, 0, -1, 0, 1, 1, -1], [1, 1, 1, -1, 0, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1], [-1, 1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 1, 1, 0, -1, -1, 1, 0, 1, 1], [1, -1, 1, -1, 1, 1, 0, -1, -1, -1, -1, 0, 1, 1, 0, -1, 1, -1, -1, 1], [-1, -1, 0, -1, 0, -1, 0, -1, 0, 0, -1, -1, 1, 0, 1, 1, 0, -1, 1, -1], [0, 0, -1, -1, -1, 1, -1, 1, -1, 1, 0, 0, 0, -1, 0, 1, -1, 1, -1, -1], [-1, -1, 0, 0, 0, -1, 1, -1, -1, 1, 0, 0, 0, 1, 1, -1, 0, 0, -1, 1], [1, -1, -1, -1, 1, 0, 0, -1, 0, 1, 0, 0, 1, -1, 1, -1, -1, 0, 0, -1], [0, 1, 1, -1, -1, 0, 1, -1, 1, 1, 1, 1, -1, -1, -1, 0, 0, 1, -1, 1], [0, 0, 0, 0, -1, 1, 0, 0, 1, -1, 0, 1, 1, -1, 0, 1, 1, 0, 0, 0], [-1, 0, 1, 1, 0, 0, 1, 1, -1, -1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1], [-1, 0, -1, -1, 1, 0, 1, -1, -1, -1, 0, 1, 0, -1, 0, -1, -1, -1, -1, 1], [0, -1, -1, -1, 1, 0, 0, 0, 1, 1, 0, 0, -1, 1, -1, -1, 1, -1, 1, -1], [-1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, -1, 1, 0, -1, -1, 1, -1], [1, 1, 0, 1, -1, -1, -1, -1, -1, 0, 1, -1, 0, 1, 0, -1, 0, 0, -1, 0], [0, 0, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 0, -1, 1, -1, 0, -1, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, -1, 1, 0, 1, 1, -1, -1, -1, 1, -1], [-1, -1, 1, -1, -1, 1, 0, 1, -1, 1, 1, 0, -1, 0, 0, 1, 0, -1, 1, 0], [1, 1, 1, 0, 0, -1, 1, -1, -1, -1, -1, 0, 1, 1, 0, 1, -1, -1, 1, 0]]


surface_b_list = [[0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, -1.0, -1.0, 1.0], [-1.0, -1.0, -1.0, 1.0, 0.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0], [1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 0.0, -1.0, -1.0], [-1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, -1.0, 1.0, 1.0, -1.0], [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, -1.0, -1.0, 0.0, 1.0, -1.0, 1.0], [0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 1.0, -1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0], [-1.0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 1.0], [0.0, -1.0, -1.0, 1.0, 1.0, 0.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0], [1.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 0.0, -1.0, 1.0, -1.0, 1.0, -1.0], [1.0, 0.0, 1.0, 1.0, -1.0, 0.0, -1.0, 1.0, 1.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0], [0.0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 0.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 1.0], [-1.0, -1.0, 0.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, -1.0, 0.0, -1.0, -1.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, -1.0, 0.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0], [1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 0.0, -1.0, 1.0, -1.0, -1.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0, -1.0, 0.0], [-1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, -1.0, 1.0, 1.0, -1.0, 0.0]]






surface_a = np.array(surface_a_list)
surface_b = np.array(surface_b_list)

print(surface_a)
print(surface_b)
print(get_offset_force_matrix(surface_a, surface_b, WIDTH))
print(get_force_values(surface_a, surface_b, WIDTH)[0], get_force_values(surface_a, surface_b, WIDTH)[1])


fig, axes = plt.subplots(1, 3, figsize=(10, 5))
axes[0].imshow(surface_a, cmap='gray')
axes[0].set_title('Surface A')
axes[1].imshow(surface_b, cmap='gray')
axes[1].set_title('Surface B')
full_offset_force_matrix = np.vstack((np.hstack((get_offset_force_matrix(surface_a, surface_b, WIDTH),
                                                 get_offset_force_matrix(np.rot90(surface_a, k=1), surface_b, WIDTH))),
                                      np.hstack((get_offset_force_matrix(np.rot90(surface_a, k=2), surface_b, WIDTH),
                                                 get_offset_force_matrix(np.rot90(surface_a, k=3), surface_b, WIDTH)))))
axes[2].imshow(full_offset_force_matrix, cmap='gray')
axes[2].set_title('Offset Force Matrices')

for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()