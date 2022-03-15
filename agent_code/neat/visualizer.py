import neat
import pygame
import sys

sys.path.append("..\\..\\.")

pygame.init()
display = pygame.display.set_mode([1000, 1000])


def main():
    pop = neat.load("pickle")
    genome = pop.population[int(sys.argv[1])]
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        display.fill((0, 0, 0))
        genome.visualize(display)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
