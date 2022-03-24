import pygame


def visualize(genome):
    pygame.init()
    display = pygame.display.set_mode([1000, 1000])
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        display.fill((0, 0, 0))
        genome.visualize(display)
        pygame.display.flip()

    pygame.quit()
