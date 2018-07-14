from math import sqrt
from random import shuffle, randrange
from time import sleep

import numpy as np
from PIL import Image
from tqdm import tqdm

c_path = (0.0, ' ', [255, 255, 255])
c_wall = (1.0, '#', [0, 0, 0])
c_start = (2.0, '&', [0, 255, 0])
c_end = (3.0, '!', [0, 255, 255])
c_visited = (5.0, '.', [255, 0, 0])
c_finalpath = (6.0, '*', [255, 255, 0])
c_current = (7.0, '@', [0, 255, 255])

codes = [c_path, c_wall, c_start, c_end, c_visited, c_finalpath, c_current]


def make_maze(w: int = 200, h: int = 200, start: bool = False, end: bool = False) -> str:
    """Creates a maze of the given width and height, with options
       to randomly set start and end locations for the exploration procedure.

       Shamelesly taken from http://stackoverflow.com/a/31634316/1245214

       :param w: width of the maze
       :param h: height of the maze
       :param start: whether to set a start position in the maze for the search algorithms
       :param end: whether to set an end position in the maze for the search algorithms
       :return: a maze in its string representation
    """
    h += 1
    w += 1
    mat = np.zeros((h, w))

    unvisited = set()
    visiting = []
    for x in [y for y in range(w) if y % 2 != 0]:
        for y in [x for x in range(h) if x % 2 != 0]:
            mat[y, x] = 1
            unvisited.add((y, x))

    pbar = tqdm(total=len(unvisited), desc='Generating DFS maze')
    while unvisited:
        pbar.update()
        current = unvisited.pop()
        unvisited_neighbors = [
            n for n in [(current[0] - 2,
                         current[1]), (current[0] + 2, current[1]), (
                            current[0], current[1] - 2), (current[0],
                                                          current[1] + 2)]
            if n in unvisited and n not in visiting
        ]

        if unvisited_neighbors:
            shuffle(unvisited_neighbors)
            visiting += current
            chosen = unvisited_neighbors[0]
            if chosen[0] + 2 == current[0]:
                mat[current[0] - 1, current[1]] = 1
            elif chosen[0] - 2 == current[0]:
                mat[current[0] + 1, current[1]] = 1
            elif chosen[1] + 2 == current[1]:
                mat[current[0], current[1] - 1] = 1
            elif chosen[1] - 2 == current[1]:
                mat[current[0], current[1] + 1] = 1

        elif visiting:
            visiting.pop()

    mat[:, 1] = 1
    mat[:, w - 2] = 1
    mat[1, :] = 1
    mat[h - 2, :] = 1
    mat = mat[1:h - 1, 1:w - 1]

    maze = array_to_string_maze(mat)

    if start is not None or end is not None:
        spaces = [i for i, c in enumerate(maze) if c == ' ']
        if start:
            i = randrange(len(spaces))
            maze = list(maze)
            maze[spaces[i]] = c_start[1]
            maze = ''.join(maze)
            del spaces[i]
        if end:
            maze = list(maze)
            maze[spaces[randrange(len(spaces))]] = c_end[1]
            maze = ''.join(maze)

    return maze


def string_to_array_maze(maze: str) -> np.ndarray:
    """Transforms a maze from its string representation into a numpy.ndarray,
       using the numerical codes set by the package.

       :param maze: a maze in its string representation
       :return: a maze in its numpy.ndarray representation
    """
    lines = maze.split('\n')
    mat = np.zeros((len(lines), len(lines[0])))
    for y, line in enumerate(lines):
        for x, c in enumerate(line):
            for code in codes:
                if c == code[1]:
                    mat[y, x] = code[0]
    return mat


def array_to_string_maze(mat: np.ndarray) -> str:
    """Transforms a maze from its numpy.ndarray representation into a string.
       Good for saving the maze into a file or printing it in the terminal.

       :param mat: a maze in its numpy.ndarray representation
       :return: a maze in its string representation
    """
    maze = ''
    for y, line in enumerate(mat):
        for x, c in enumerate(line):
            for code in codes:
                if c == code[0]:
                    maze += code[1]
                    break
        maze += '\n'
    return maze


def array_to_color_codes_maze(mat: np.ndarray) -> np.ndarray:
    """Transforms a maze from its numpy.ndarray representation into a list of 8-bit color codes.
       Good for saving the maze into an image using PIL, for example.

       :param mat: a maze in its numpy.ndarray representation
       :return: a numpy.ndarray containing the color codes for each tile in the maze
    """
    img = np.zeros((mat.shape[0], mat.shape[1], 3), dtype=np.uint8)
    for code in codes:
        img[mat == code[0]] = code[2]
    return img


def distance(p1: (int, int), p2: (int, int)) -> float:
    """Calculates the Euclidean distance between two cartesian coordinates.

    :param p1: a tuple containing (y, x) positions of the first coordinate
    :param p2: a tuple containing (y, x) positions of the second coordinate
    :return: Euclidean distance between p1 and p2
    """
    (y2, x2), (y1, x1) = p2, p1
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def solve(maze: str, strategy: str = 'dfs', verbosity: int = 0, file: str = None):
    """Solves a maze using one of the available algorithms.

    :param maze: a maze in its string representation
    :param strategy: which algorithm to use. Either 'dfs', 'bfs' or 'a*'
    :param verbosity: verbosity level of the exploration procedure. From 0 to 3.
    :param file: Path to save an image file of the exploration procedure.
    """
    if strategy not in ['dfs', 'bfs', 'a*']:
        raise ValueError('Invalid option for strategy')

    mat = string_to_array_maze(maze)

    paths = np.sum(mat == 0)

    start = np.nonzero(mat == c_start[0])
    start = (start[0][0], start[1][0])

    end = np.nonzero(mat == c_end[0])
    end = (end[0][0], end[1][0])

    current = start

    stack = [current]

    step = 0
    explored = 0
    walked = 0
    max_depth = 0

    pbar = tqdm(total=paths)

    while stack:
        if strategy == 'bfs':
            current = stack.pop(0)
        else:
            current = stack.pop()

        y, x = current

        step += 1
        mat[y, x] = c_current[0]

        if step > max_depth:
            max_depth = step

        explored += 1
        walked += 1

        if verbosity >= 2:
            print(
                '{0}Visiting {1}...\nExplored {2} tiles ({7}%).\nWalked {3} tiles.\nDepth {4} (max {5})\n{6}'.
                    format(
                    chr(27) + "[2J", current, explored, walked, step,
                    max_depth, array_to_string_maze(mat),
                    round(explored / paths * 100, 4)))
            sleep(.1)
        elif verbosity >= 1:
            pbar.update()
        if current == end:
            mat[start[0], start[1]] = c_start[0]
            mat[y, x] = c_end[0]
            if verbosity >= 1:
                print(chr(27) + "[2J")
            print(
                'Exit at {0}.\nExplored {1} tiles ({2}%).\nWalked {3} tiles.\nDepth {4} (max {5})'.
                    format(current, explored, round(explored / paths * 100, 4),
                           walked, step, max_depth))
            if verbosity >= 1:
                print(array_to_string_maze(mat))

            if file is not None:
                Image.fromarray(array_to_color_codes_maze(mat), 'RGB').resize(
                    (1024, 1024)).save(file)
            break
        else:
            mat[y, x] = c_visited[0]

            next = []

            for prospective_next in [(y - 1, x), (y, x - 1), (y + 1, x),
                                     (y, x + 1)]:
                if mat[prospective_next[0], prospective_next[1]] not in [
                    c_wall[0], c_visited[0]
                ]:
                    next.append(prospective_next)

            if strategy == 'a*':
                next = sorted(next, key=lambda p: distance(p, end))
            else:
                shuffle(next)

            stack += next

            walked += 1
            if verbosity >= 3:
                mat[y, x] = c_current[0]
                print(
                    '{0}Visiting {1}...\nExplored {2} tiles ({7}%).\nWalked {3} tiles.\nDepth {4} (max {5})\n{6}'.
                        format(
                        chr(27) + "[2J", current, explored, walked, step,
                        max_depth, array_to_string_maze(mat),
                        round(explored / paths * 100, 4)))
                sleep(.1)
