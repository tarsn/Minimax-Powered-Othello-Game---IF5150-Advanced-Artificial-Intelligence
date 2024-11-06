from othello_game import OthelloGame
import math
import random


def get_best_move(game, max_depth=8):
    """
    Given the current game state, this function returns the best move for the AI player using the Alpha-Beta Pruning
    algorithm with a specified maximum search depth.

    Parameters:
        game (OthelloGame): The current game state.
        max_depth (int): The maximum search depth for the Alpha-Beta algorithm.

    Returns:
        tuple: A tuple containing the evaluation value of the best move and the corresponding move (row, col).
    """
    if len(game.ai_mode) <= 17:
        if game.ai_mode == "minimax_1":
            _, best_move = alphabeta(game, 2)
        elif game.ai_mode == "minimax_2":
            _, best_move = alphabeta(game, 3)
        elif game.ai_mode == "minimax_3":
            _, best_move = alphabeta(game, 4)
        elif game.ai_mode == "local_search":
            best_move = local_search(game)
        else: 
            best_move = genetic(game)
    else:
        algos = game.ai_mode.split("_vs_")
        if game.current_player == 1:
            if algos[0] == "minimax_1":
                _, best_move = alphabeta(game, 2)
            elif algos[0] == "minimax_2":
                _, best_move = alphabeta(game, 3)
            elif algos[0] == "minimax_3":
                _, best_move = alphabeta(game, 4)
            elif algos[0] == "local_search":
                best_move = local_search(game)
            else: 
                best_move = genetic(game)
        else:
            if algos[1] == "minimax_1":
                _, best_move = alphabeta(game, 2)
            elif algos[1] == "minimax_2":
                _, best_move = alphabeta(game, 3)
            elif algos[1] == "minimax_3":
                _, best_move = alphabeta(game, 4)
            elif algos[1] == "local_search":
                best_move = local_search(game)
            else: 
                best_move = genetic(game)
    return best_move


def alphabeta(
    game, max_depth, maximizing_player=True, alpha=float("-inf"), beta=float("inf")
):
    """
    Alpha-Beta Pruning algorithm for selecting the best move for the AI player.

    Parameters:
        game (OthelloGame): The current game state.
        max_depth (int): The maximum search depth for the Alpha-Beta algorithm.
        maximizing_player (bool): True if maximizing player (AI), False if minimizing player (opponent).
        alpha (float): The alpha value for pruning. Defaults to negative infinity.
        beta (float): The beta value for pruning. Defaults to positive infinity.

    Returns:
        tuple: A tuple containing the evaluation value of the best move and the corresponding move (row, col).
    """
    if max_depth == 0 or game.is_game_over():
        return evaluate_game_state(game), None

    valid_moves = game.get_valid_moves()

    if maximizing_player:
        max_eval = float("-inf")
        best_move = None

        for move in valid_moves:
            new_game = OthelloGame(player_mode=game.player_mode)
            new_game.board = [row[:] for row in game.board]
            new_game.current_player = game.current_player
            new_game.make_move(*move)

            eval, _ = alphabeta(new_game, max_depth - 1, False, alpha, beta)

            if eval > max_eval:
                max_eval = eval
                best_move = move

            alpha = max(alpha, eval)
            if beta <= alpha:
                break

        return max_eval, best_move
    else:
        min_eval = float("inf")
        best_move = None

        for move in valid_moves:
            new_game = OthelloGame(player_mode=game.player_mode)
            new_game.board = [row[:] for row in game.board]
            new_game.current_player = game.current_player
            new_game.make_move(*move)

            eval, _ = alphabeta(new_game, max_depth - 1, True, alpha, beta)

            if eval < min_eval:
                min_eval = eval
                best_move = move

            beta = min(beta, eval)
            if beta <= alpha:
                break

        return min_eval, best_move


def evaluate_game_state(game):
    """
    Evaluates the current game state for the AI player.

    Parameters:
        game (OthelloGame): The current game state.

    Returns:
        float: The evaluation value representing the desirability of the game state for the AI player.
    """
    # Evaluation weights for different factors
    coin_parity_weight = 1.0
    mobility_weight = 2.0
    corner_occupancy_weight = 5.0
    stability_weight = 3.0
    edge_occupancy_weight = 2.5

    # Coin parity (difference in disk count)
    player_disk_count = sum(row.count(game.current_player) for row in game.board)
    opponent_disk_count = sum(row.count(-game.current_player) for row in game.board)
    coin_parity = player_disk_count - opponent_disk_count

    # Mobility (number of valid moves for the current player)
    player_valid_moves = len(game.get_valid_moves())
    opponent_valid_moves = len(
        OthelloGame(player_mode=-game.current_player).get_valid_moves()
    )
    mobility = player_valid_moves - opponent_valid_moves

    # Corner occupancy (number of player disks in the corners)
    corner_occupancy = sum(
        game.board[i][j] for i, j in [(0, 0), (0, 7), (7, 0), (7, 7)]
    )

    # Stability (number of stable disks)
    stability = calculate_stability(game)

    # Edge occupancy (number of player disks on the edges)
    edge_occupancy = sum(game.board[i][j] for i in [0, 7] for j in range(1, 7)) + sum(
        game.board[i][j] for i in range(1, 7) for j in [0, 7]
    )

    # Combine the factors with the corresponding weights to get the final evaluation value
    evaluation = (
        coin_parity * coin_parity_weight
        + mobility * mobility_weight
        + corner_occupancy * corner_occupancy_weight
        + stability * stability_weight
        + edge_occupancy * edge_occupancy_weight
    )

    return evaluation


def evaluate_game_state_v2(game):
    # Tentukan tahapan permainan berdasarkan persentase bidak yang sudah ditempatkan di papan
    total_disks = sum(row.count(1) + row.count(-1) for row in game.board)
    board_size = len(game.board) * len(game.board[0])
    game_stage_ratio = total_disks / board_size

    # Tentukan bobot untuk setiap faktor berdasarkan tahapan permainan
    if game_stage_ratio < 0.2:  # Early game
        coin_parity_weight = 0.5
        mobility_weight = 3.0
        corner_occupancy_weight = 5.0
        stability_weight = 2.0
        edge_occupancy_weight = 2.0
        aggressiveness_weight = 1.0
    elif game_stage_ratio < 0.8:  # Mid game
        coin_parity_weight = 1.0
        mobility_weight = 2.5
        corner_occupancy_weight = 4.0
        stability_weight = 3.0
        edge_occupancy_weight = 2.5
        aggressiveness_weight = 1.5
    else:  # Late game
        coin_parity_weight = 2.0
        mobility_weight = 1.0
        corner_occupancy_weight = 6.0
        stability_weight = 4.0
        edge_occupancy_weight = 3.0
        aggressiveness_weight = 0.5

    # Coin parity
    player_disk_count = sum(row.count(game.current_player) for row in game.board)
    opponent_disk_count = sum(row.count(-game.current_player) for row in game.board)
    coin_parity = player_disk_count - opponent_disk_count

    # Mobility
    player_valid_moves = len(game.get_valid_moves())
    opponent_valid_moves = len(
        OthelloGame(player_mode=-game.current_player).get_valid_moves()
    )
    mobility = player_valid_moves - opponent_valid_moves

    # Corner occupancy
    corner_occupancy = sum(
        game.board[i][j] for i, j in [(0, 0), (0, 7), (7, 0), (7, 7)]
    )

    # Stability
    stability = calculate_stability(game)

    # Edge occupancy
    edge_occupancy = sum(game.board[i][j] for i in [0, 7] for j in range(1, 7)) + sum(
        game.board[i][j] for i in range(1, 7) for j in [0, 7]
    )

    # Aggressiveness: jumlah disk yang dekat dengan lawan
    aggressiveness = 0
    for i in range(8):
        for j in range(8):
            if game.board[i][j] == game.current_player:
                # Hitung bidak lawan di sekitar bidak pemain saat ini
                adjacent_opponent_count = sum(
                    1 for x, y in [
                        (i-1, j), (i+1, j), (i, j-1), (i, j+1),
                        (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)
                    ]
                    if 0 <= x < 8 and 0 <= y < 8 and game.board[x][y] == -game.current_player
                )
                aggressiveness += adjacent_opponent_count

    # Hitung evaluasi akhir dengan bobot yang telah disesuaikan
    evaluation = (
        coin_parity * coin_parity_weight
        + mobility * mobility_weight
        + corner_occupancy * corner_occupancy_weight
        + stability * stability_weight
        + edge_occupancy * edge_occupancy_weight
        - aggressiveness * aggressiveness_weight
    )

    return evaluation


def calculate_stability(game):
    """
    Calculates the stability of the AI player's disks on the board.

    Parameters:
        game (OthelloGame): The current game state.

    Returns:
        int: The number of stable disks for the AI player.
    """

    def neighbors(row, col):
        return [
            (row + dr, col + dc)
            for dr in [-1, 0, 1]
            for dc in [-1, 0, 1]
            if (dr, dc) != (0, 0) and 0 <= row + dr < 8 and 0 <= col + dc < 8
        ]

    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    edges = [(i, j) for i in [0, 7] for j in range(1, 7)] + [
        (i, j) for i in range(1, 7) for j in [0, 7]
    ]
    inner_region = [(i, j) for i in range(2, 6) for j in range(2, 6)]
    regions = [corners, edges, inner_region]

    stable_count = 0

    def is_stable_disk(row, col):
        return (
            all(game.board[r][c] == game.current_player for r, c in neighbors(row, col))
            or (row, col) in edges + corners
        )

    for region in regions:
        for row, col in region:
            if game.board[row][col] == game.current_player and is_stable_disk(row, col):
                stable_count += 1

    return stable_count


def local_search(game):
    valid_moves = game.get_valid_moves()
    best_move = None
    best_value = -9999999
    
    for move in valid_moves:

        old_game = OthelloGame(player_mode=game.player_mode)
        old_game.board = [row[:] for row in game.board]
        old_game.current_player = game.current_player
        old_game.ai_mode = game.ai_mode

        T = 100
        best_current_value = -9999
        for i in range(200):
            old_valid_moves = old_game.get_valid_moves()
            if len(old_valid_moves) == 0:
                break
            index = math.floor(random.random()*len(old_valid_moves))

            new_game = OthelloGame(player_mode=old_game.player_mode)
            new_game.board = [row[:] for row in old_game.board]
            new_game.current_player = old_game.current_player
            new_game.ai_mode = old_game.ai_mode
            new_game.make_move(*old_valid_moves[index])

            new_value = None
            old_value = None
            if old_game.current_player == game.current_player:
                new_value = evaluate_game_state(new_game)
                old_game.current_player = -1*old_game.current_player
                old_value = evaluate_game_state(old_game)
                old_game.current_player = -1*old_game.current_player
            else:
                new_game.current_player = -1*new_game.current_player
                new_value = evaluate_game_state(new_game)
                new_game.current_player = -1*new_game.current_player
                old_value = evaluate_game_state(old_game)
            deltaE = new_value - old_value
            if deltaE > 0:
                best_current_value = new_value
                old_game = OthelloGame(player_mode=new_game.player_mode)
                old_game.board = [row[:] for row in new_game.board]
                old_game.current_player = new_game.current_player
                old_game.ai_mode = new_game.ai_mode
            else:
                if T == 0:
                    break
                elif math.exp(deltaE/T) > random.random():
                    old_game = OthelloGame(player_mode=new_game.player_mode)
                    old_game.board = [row[:] for row in new_game.board]
                    old_game.current_player = new_game.current_player
                    old_game.ai_mode = new_game.ai_mode
            
            T *= 0.98

        if best_current_value > best_value:
            best_value = best_current_value
            best_move = move
    
    return best_move

def genetic(game):
    best_move = None
    valid_moves = game.get_valid_moves()
    if len(valid_moves) == 0:
        return best_move
    random.shuffle(valid_moves)
    length_genetic = int(math.floor(math.log2(len(valid_moves))))

    binary_strings = []
    values = []
    for i in range(0, int(math.pow(2, length_genetic))):
        binary_string = bin(i)[2:]  
        padded_binary_string = binary_string.zfill(length_genetic) 

        new_game = OthelloGame(player_mode=game.player_mode)
        new_game.board = [row[:] for row in game.board]
        new_game.current_player = game.current_player
        new_game.ai_mode = game.ai_mode
        new_game.make_move(*valid_moves[i])

        new_value = evaluate_game_state(new_game)
        new_valid_moves = new_game.get_valid_moves()
        random.shuffle(new_valid_moves)

        if(len(new_valid_moves) >= 2):
            for j in range(2):
                new_child_game = OthelloGame(player_mode=new_game.player_mode)
                new_child_game.board = [row[:] for row in new_game.board]
                new_child_game.current_player = new_game.current_player
                new_child_game.ai_mode = new_game.ai_mode
                new_child_game.make_move(*new_valid_moves[j])

                new_child_game.current_player = -1*new_child_game.current_player
                binary_strings.append(f"{padded_binary_string}{j}")
                values.append((new_value+evaluate_game_state(new_child_game))/2)
                new_child_game.current_player = -1*new_child_game.current_player
        else:
            binary_strings.append(f"{padded_binary_string}0")
            binary_strings.append(f"{padded_binary_string}1")
            if len(new_valid_moves) == 1:
                new_child_game = OthelloGame(player_mode=new_game.player_mode)
                new_child_game.board = [row[:] for row in new_game.board]
                new_child_game.current_player = new_game.current_player
                new_child_game.ai_mode = new_game.ai_mode
                new_child_game.make_move(*new_valid_moves[j])
                new_child_game.current_player = -1*new_child_game.current_player
                values.append((new_value + evaluate_game_state(new_child_game))/2)
            else:
                values.append(new_value/2)
            values.append(new_value/2)

    if len(binary_string) <= 2:
        return valid_moves[0]

    sorted_lists = sorted(zip(values, binary_strings), key=lambda x: x[0], reverse=True)
    list1_sorted, list2_sorted = zip(*sorted_lists)

    values = list(list1_sorted)
    binary_strings = list(list2_sorted)

    total = sum(values)
    percentages = [(num / total) for num in values]

    indices = []
    for i in range(4):
        rand = random.random()
        j = 0
        tmpPrecentage = percentages[0]
        while tmpPrecentage < rand and j < len(percentages):
            tmpPrecentage += percentages[j]
            j += 1
        indices.append(j)

    individu1 = f"{binary_strings[indices[0]][:length_genetic//2]}{binary_strings[indices[1]][length_genetic//2:]}"
    individu1 = individu1[:length_genetic//2] + str(random.randint(0,1)) + individu1[length_genetic//2+1:]
    individu2 = f"{binary_strings[indices[0]][:length_genetic//2]}{binary_strings[indices[1]][length_genetic//2:]}"
    individu2 = individu2[:length_genetic//2] + str(random.randint(0,1)) + individu2[length_genetic//2+1:]
        
    try:
        index1 = binary_strings.index(individu1)
        index2 = binary_strings.index(individu2)
    except:
        return valid_moves[0]

    if values[index1] > values[index2]:
        best_move = valid_moves[int(individu1[:length_genetic], 2)]
    else:
        best_move = valid_moves[int(individu2[:length_genetic], 2)]

    return best_move
