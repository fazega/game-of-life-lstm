"""Visualizes the environment and the agents playing in it."""

import haiku as hk
import numpy as np
import pygame

import agent as agent_lib
import env as env_lib
import grid as grid_lib
import network as network_lib


_EGOCENTRIC_SIZE = 5
_SCREEN_SIZE = 600


def visualize(
    env_state: env_lib.EnvironmentState,
    model_params: hk.Params,
) -> None:
    """Displays some agents interacting in the environment."""
    rng = np.random.default_rng(1)
    model = hk.transform(network_lib.policy)
    size_grid = len(env_state.grid)

    # Initialize the pygame screen.
    pygame.init()  # pylint: disable=no-member
    screen = pygame.display.set_mode((_SCREEN_SIZE, _SCREEN_SIZE))
    pygame.display.set_caption("Life game")
    my_font = pygame.font.SysFont("Arial", 16, True)

    # Used for logging.
    last_red_scores, last_blue_scores = [], []

    running = True
    while running:
        # Event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # pylint: disable=no-member
                running = False

        gridsize = _SCREEN_SIZE // size_grid
        for i in range(size_grid):
            for j in range(size_grid):
                pixel = env_state.grid[i, j]
                if pixel == 0:
                    color = (0, 0, 0)
                elif pixel == grid_lib.Side.RED:
                    color = (255, 0, 0)
                elif pixel == grid_lib.Side.BLUE:
                    color = (0, 0, 255)
                pygame.draw.rect(
                    screen,
                    color,
                    (j * gridsize, i * gridsize, gridsize, gridsize),
                )

        for agent_type, agent_pos in zip(
            env_state.agent_types, env_state.agent_positions
        ):
            x, y = agent_pos
            color = (
                (255, 0, 0) if agent_type == grid_lib.Side.RED else (0, 0, 255)
            )
            pygame.draw.circle(
                screen,
                color,
                (y * gridsize + gridsize / 2, x * gridsize + gridsize / 2),
                gridsize // 2,
                2,
            )

        # Update.
        # Get the egocentric views of the agents.
        views = agent_lib.egocentric_views(
            agent_positions=env_state.agent_positions,
            grid=env_state.grid,
            size=_EGOCENTRIC_SIZE,
        )

        # Retrieve the policies and sample random actions.
        log_probs, new_states = model.apply(
            model_params, None, views, env_state.agent_states
        )
        probs = np.exp(log_probs)
        oh_actions = rng.multinomial(n=1, pvals=probs, size=(len(log_probs),))
        actions = np.argmax(oh_actions, axis=-1)

        # Update the environment with the actions.
        env_state.update_agent_positions(actions)
        env_state.grid.update_swap(
            actions, env_state.agent_types, env_state.agent_positions
        )
        env_state.grid.update_gol()

        # Display the scores.
        red_sum = float(np.sum(env_state.grid == 1))
        blue_sum = float(np.sum(env_state.grid == -1))
        red_score = red_sum / (red_sum + blue_sum)
        blue_score = blue_sum / (red_sum + blue_sum)
        max_scores = 30
        delta = 4
        height = 60
        last_red_scores.append(red_score)
        if len(last_red_scores) >= max_scores:
            last_red_scores.pop(0)
        last_blue_scores.append(blue_score)
        if len(last_blue_scores) >= max_scores:
            last_blue_scores.pop(0)

        pygame.draw.rect(
            screen,
            (0, 0, 0),
            (_SCREEN_SIZE - 250, 10, delta * max_scores, height),
        )
        pygame.draw.rect(
            screen,
            (255, 255, 255),
            (_SCREEN_SIZE - 250, 10, delta * max_scores, height),
            width=2,
        )
        if len(last_red_scores) >= 2:
            red_scores_coords = [
                (_SCREEN_SIZE - 250 + i * delta, 10 + height * (1 - score))
                for i, score in enumerate(last_red_scores)
            ]
            pygame.draw.lines(
                screen, (255, 0, 0), False, red_scores_coords, width=2
            )
        if len(last_blue_scores) >= 2:
            blue_scores_coords = [
                (_SCREEN_SIZE - 250 + i * delta, 10 + height * (1 - score))
                for i, score in enumerate(last_blue_scores)
            ]
            pygame.draw.lines(
                screen, (0, 0, 255), False, blue_scores_coords, width=2
            )
        red_text = my_font.render(
            f"Red: {red_score*100:.1f}%", False, (255, 255, 255)
        )
        screen.blit(red_text, (_SCREEN_SIZE - 100, 10))
        blue_text = my_font.render(
            f"Blue: {blue_score*100:.1f}%", False, (255, 255, 255)
        )
        screen.blit(blue_text, (_SCREEN_SIZE - 100, 40))
        pygame.display.update()

        # Cap the frame rate.
        pygame.time.Clock().tick(20)

    # Clean up.
    pygame.quit()  # pylint: disable=no-member
