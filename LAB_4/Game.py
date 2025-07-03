#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

#budowanie siatki
def create_grid(rows, cols):
    return [[0]*cols for _ in range(rows)]

#zliczanie zywych sasiadow
def neighbour_count(grid, x, y):
    alive = 0
    rows, cols = len(grid), len(grid[0])
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = (x + dx) % rows
            ny = (y + dy) % cols
            alive += grid[nx][ny]
    return alive

#nowa siatka po zastosowaniu warunkow
def transition_rules(grid):
    rows, cols = len(grid), len(grid[0])
    new = [row[:] for row in grid]
    for x in range(rows):
        for y in range(cols):
            alive = neighbour_count(grid, x, y)
            if grid[x][y] == 1:
                new[x][y] = 1 if 2 <= alive <= 3 else 0
            else:
                new[x][y] = 1 if alive == 3 else 0
    return new

#ile zycywch komorek w siatce
def count_alive_cells(grid):
    return sum(sum(row) for row in grid)

#nakladanie siatki 
def insert_pattern(grid, pattern, x, y):
    if pattern == "block":
        coords = [(0,0),(0,1),(1,0),(1,1)]
    elif pattern == "beehive":
        coords = [(0,1),(0,2),(1,0),(1,3),(2,1),(2,2)]
    elif pattern == "boat":
        coords = [(0,0),(0,1),(1,0),(1,2),(2,1)]
    elif pattern == "blinker":
        coords = [(0,0),(0,1),(0,2)]
    elif pattern == "toad":
        coords = [(0,1),(0,2),(0,3),(1,0),(1,1),(1,2)]
    else:
        raise ValueError(f"Nieznany wzorzec: {pattern}")
    for dx, dy in coords:
        grid[x+dx][y+dy] = 1

def still_life(grid):
    insert_pattern(grid, "block",   5,  5)
    insert_pattern(grid, "beehive",20, 20)
    insert_pattern(grid, "boat",   35, 35)

def oscylator(grid):
    insert_pattern(grid, "blinker",5,  5)
    insert_pattern(grid, "toad",   20, 20)

def glider_gun(grid, x, y):
    coords = [
        (0,24),(1,22),(1,24),(2,12),(2,13),(2,20),(2,21),(2,34),(2,35),
        (3,11),(3,15),(3,20),(3,21),(3,34),(3,35),(4,0),(4,1),(4,10),
        (4,16),(4,20),(4,21),(5,0),(5,1),(5,10),(5,14),(5,16),(5,17),
        (5,22),(5,24),(6,10),(6,16),(6,24),(7,11),(7,15),(8,12),(8,13)
    ]
    for dx, dy in coords:
        grid[x+dx][y+dy] = 1

def matuzalech(grid, x, y):
    coords = [(0,1),(0,2),(1,0),(1,1),(2,1)]
    for dx, dy in coords:
        grid[x+dx][y+dy] = 1

def animate(grid, steps, interval=200, save_path=None):
    fig, ax = plt.subplots()
    img  = ax.imshow(grid, cmap='binary')
    text = ax.text(0.95, 0.95, '', transform=ax.transAxes,
                   fontsize=12, color='red', ha='right', va='top',
                   bbox=dict(facecolor='white', alpha=0.7))

    def update(frame):
        nonlocal grid
        grid = transition_rules(grid)
        img.set_data(grid)
        alive = count_alive_cells(grid)
        ax.set_title(f"Epoka: {frame+1}")
        text.set_text(f"Żywych: {alive}")
        if frame+1 == steps:
            ani.event_source.stop()
        return img, text

    ani = animation.FuncAnimation(
        fig, update, frames=steps, interval=interval, blit=False
    )

    if save_path:
        writer = PillowWriter(fps=1000/interval)
        ani.save(save_path, writer=writer)
        print(f"Zapisano animację do: {save_path}")
        plt.close(fig)
    else:
        plt.show()

def main():
    actions = {
        "1": ("wzorce ciągłego życia", still_life, 5,  "still_life.gif"),
        "2": ("oscylatory",          oscylator, 10, "oscylator.gif"),
        "3": ("działo Glider Gun",   glider_gun,100,"glider_gun.gif"),
        "4": ("Matuzalech",           matuzalech, 250,"matuzalech.gif")
    }
    while True:
        choice = input(
            "Wybierz symulację:\n"
            " 1 – wzorce ciągłego życia\n"
            " 2 – oscylatory\n"
            " 3 – działo Glider Gun\n"
            " 4 – Matuzalech\n"
            " end – zakończ\n> "
        ).strip().lower()
        if choice == "end":
            print("Koniec programu.")
            break
        if choice not in actions:
            print("Nieznany tryb, spróbuj ponownie.")
            continue

        desc, func, steps, fname = actions[choice]
        grid = create_grid(50, 50)
        if choice in ("3", "4"):
            func(grid, 10, 10)
        else:
            func(grid)

        print(f"Symulacja {desc}. Początkowe żywe komórki: {count_alive_cells(grid)}")
        animate(grid, steps, interval=200, save_path=fname)

if __name__ == "__main__":
    main()
