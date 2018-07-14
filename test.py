from maze_runner import make_maze, solve

if __name__ == '__main__':
    maze = make_maze(start=True, end=True)
    solve(maze, strategy='bfs', verbosity=1, file='sphinx/bfs.png')
    solve(maze, strategy='dfs', verbosity=1, file='sphinx/dfs.png')
    solve(maze, strategy='a*', verbosity=1, file='sphinx/h.png')
