from neat import *
import pygame

pygame.init()
display = pygame.display.set_mode([500, 500])

g = Population.Genome.fresh()
g.mutate(20)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    display.fill((0, 0, 0))
    g.visualize(display)
    pygame.display.flip()

pygame.quit()
