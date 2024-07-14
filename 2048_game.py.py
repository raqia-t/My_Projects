import tkinter as tk
import random
import numpy as np
import copy

class Engine:
    def __init__(self):
        self.size = 4
        self.board = [[0 for i in range(self.size)] for i in range(self.size)]
        self.score = 0
        self.numMoves = 0
        self.moveList = ['down', 'left', 'up', 'right']
        self.addRandBlock()
        self.addRandBlock()

    def setBoard(self, boardString):
        boardList = list(map(int, boardString.split(' ')))
        if len(boardList) != 16:
            raise ValueError("Invalid board string. It must contain 16 space-separated integers.")
        self.board = [boardList[i:i + self.size] for i in range(0, len(boardList), self.size)]
        self.score = 0
        self.numMoves = 0

    def scoreBonus(self, val):
        score = {2: 4, 4: 8, 8: 16, 16: 32, 32: 64, 64: 128, 128: 256, 256: 512, 512: 1024, 1024: 2048, 2048: 4096,
                 4096: 8192, 8192: 16384, 16384: 32768, 32768: 65536, 65536: 131072}
        return score[val]

    def rotateBoard(self, board, count):
        rotated = [row[:] for row in board]
        for c in range(count):
            rotated = [[0 for i in range(self.size)] for i in range(self.size)]
            for row in range(self.size):
                for col in range(self.size):
                    rotated[self.size - col - 1][row] = board[row][col]
            board = rotated
        return rotated

    def makeMove(self, moveDir):
        if self.gameOver():
            return False

        board = self.board
        rotateCount = self.moveList.index(moveDir)
        moved = False
        if rotateCount:
            board = self.rotateBoard(board, rotateCount)

        merged = [[0 for i in range(self.size)] for i in range(self.size)]
        for row in range(self.size - 1):
            for col in range(self.size):
                currentTile = board[row][col]
                nextTile = board[row + 1][col]
                if not currentTile:
                    continue
                if not nextTile:
                    for x in range(row + 1):
                        board[row - x + 1][col] = board[row - x][col]
                    board[0][col] = 0
                    moved = True
                    continue
                if merged[row][col]:
                    continue
                if currentTile == nextTile:
                    if (row < self.size - 2 and nextTile == board[row + 2][col]):
                        continue
                    board[row + 1][col] *= 2
                    for x in range(row):
                        board[row - x][col] = board[row - x - 1][col]
                    board[0][col] = 0
                    merged[row + 1][col] = 1
                    self.score += self.scoreBonus(currentTile)
                    moved = True
        if rotateCount:
            board = self.rotateBoard(board, 4 - rotateCount)

        self.board = board
        if moved:
            self.numMoves += 1
            self.addRandBlock()
            return True
        else:
            return False

    def addRandBlock(self, val=None):
        avail = self.availableSpots()
        if avail:
            (row, column) = avail[random.randint(0, len(avail) - 1)]
            self.board[row][column] = 4 if random.randint(0, 9) == 9 else 2

    def availableSpots(self):
        spots = []
        for row in enumerate(self.board):
            for col in enumerate(row[1]):
                if col[1] == 0:
                    spots.append((row[0], col[0]))
        return spots

    def gameOver(self):
        if self.availableSpots():
            return False

        for move in self.moveList:
            board = self.rotateBoard(copy.deepcopy(self.board), self.moveList.index(move))
            for row in range(self.size - 1):
                for col in range(self.size):
                    currentTile = board[row][col]
                    nextTile = board[row + 1][col]
                    if not currentTile:
                        continue
                    if not nextTile:
                        return False
                    if currentTile == nextTile:
                        if (row < self.size - 2 and nextTile == board[row + 2][col]):
                            continue
                        return False
        return True

def mcts_strategy(game, num_simulations):
    average_scores = {move: 0 for move in game.moveList}

    for move in average_scores.keys():
        total_score = 0
        num_valid_simulations = 0
        for _ in range(num_simulations):
            game_copy = copy.deepcopy(game)
            original_board = copy.deepcopy(game_copy.board)
            game_copy.makeMove(move)
            if original_board == game_copy.board:
                continue

            num_valid_simulations += 1
            while not game_copy.gameOver():
                random_move = random.choice(game_copy.moveList)
                valid_move = game_copy.makeMove(random_move)
                if not valid_move:
                    break

            total_score += game_copy.score

        average_scores[move] = total_score / num_valid_simulations if num_valid_simulations > 0 else 0

    best_move = max(average_scores, key=average_scores.get)
    return best_move

class Game2048GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("2048 Game")
        self.game = Engine()

        self.board_frame = tk.Frame(self.root)
        self.board_frame.pack()

        self.score_label = tk.Label(self.root, text="Score: 0")
        self.score_label.pack()

        self.tiles = [[tk.Label(self.board_frame, text="", width=4, height=2, font=("Helvetica", 24), borderwidth=1,
                                relief="solid") for _ in range(4)] for _ in range(4)]

        for i in range(4):
            for j in range(4):
                self.tiles[i][j].grid(row=i, column=j, padx=5, pady=5)

        self.update_board()
        self.root.bind("<Key>", self.handle_keypress)

        self.ai_button = tk.Button(self.root, text="AI Move", command=self.ai_move)
        self.ai_button.pack()

    def handle_keypress(self, event):
        key = event.keysym
        move_made = False
        if key == "Up":
            move_made = self.game.makeMove("up")
        elif key == "Down":
            move_made = self.game.makeMove("down")
        elif key == "Left":
            move_made = self.game.makeMove("left")
        elif key == "Right":
            move_made = self.game.makeMove("right")

        if move_made:
            self.update_board()
            if self.game.gameOver():
                self.show_game_over()

    def update_board(self):
        color_map = {
            0: "#cdc1b4",
            2: "#eee4da",
            4: "#ede0c8",
            8: "#f2b179",
            16: "#f59563",
            32: "#f67c5f",
            64: "#f65e3b",
            128: "#edcf72",
            256: "#edcc61",
            512: "#edc850",
            1024: "#edc53f",
            2048: "#edc22e",
            4096: "#3c3a32",
            8192: "#3c3a32",
            16384: "#3c3a32",
            32768: "#3c3a32",
            65536: "#3c3a32",
        }

        for i in range(4):
            for j in range(4):
                value = self.game.board[i][j]
                self.tiles[i][j].config(text=str(value) if value != 0 else "", bg=color_map.get(value, "#3c3a32"))
        self.score_label.config(text=f"Score: {self.game.score}")

    def show_game_over(self):
        game_over_label = tk.Label(self.root, text="Game Over", font=("Helvetica", 24))
        game_over_label.pack()
        self.root.unbind("<Key>")

    def ai_move(self):
        if not self.game.gameOver():
            best_move = mcts_strategy(self.game, 100)
            self.game.makeMove(best_move)
            self.update_board()
            if self.game.gameOver():
                self.show_game_over()

root = tk.Tk()
game_gui = Game2048GUI(root)
root.mainloop()
