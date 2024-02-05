import pygame
import os
import math
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"
COLORS_TO_RGB = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'orange': (255, 165, 0),
    'purple': (200, 0, 200),
}
import random
class DataGenerator():
    
    def __init__(self, resolution=128, path=""):
        self.resolution = 128
        self.path = path
        pygame.init()
        self.screen = pygame.display.set_mode([self.resolution,resolution])
        self.colors = {
            "red":(200,40,0),
            "green": (53,194,41),
            "blue": (41,87,194),
            "orange" : (217,101,13),
            "purple": (127,41,194),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'brown': (165, 42, 42),
        }
        self.shapes = [lambda screen, color: pygame.draw.circle(screen, color, (64,64), 62), lambda screen, color: pygame.draw.rect(screen, color, (64, 64, 100, 100))]


    def generate_polygon(self, num_vertices, shape_color):

        polygon_center = (64, 64)  # Replace with the desired center coordinates
        polygon_distance = 64  # Replace with the desired distance from the center
            
        def generate_polygon_vertices(num_vertices, center, distance):
            vertices = []
            angle_increment = 2 * math.pi / num_vertices
            for i in range(num_vertices):
                angle = i * angle_increment
                x = center[0] + distance * math.cos(angle)
                y = center[1] + distance * math.sin(angle)
                vertices.append((x, y))
            return vertices
        polygon_vertices = generate_polygon_vertices(num_vertices, polygon_center, polygon_distance)
        polygon_surface = pygame.Surface((128, 128), pygame.SRCALPHA)
        # polygon_surface = pygame.transform.rotate(polygon_surface, random.randint(0,360))
        pygame.draw.polygon(polygon_surface, shape_color, polygon_vertices)
        return polygon_surface
    def generate_font_letters(self, fontName):
        '''Generates letters for the given font'''
        font = pygame.font.SysFont(fontName, self.resolution)
        for x in ALPHABET:
            self.screen.fill((0,0,0)) # Fill screen with black
            letter = font.render(x, True, (255,255,255), (0,0,0))
            textRect = letter.get_rect()
            textRect.center = (self.resolution // 2, self.resolution // 2)
            self.screen.blit(letter, textRect)
            pygame.display.flip()
            pygame.image.save(self.screen, f"./fonts/{fontName}/{x}.jpg")
    

    def generate_photos(self, font, distribution, images_per_color, startNum = 0):
        with open(f".{self.path}/dataset/labels.txt", "a") as file:
            currentDeg = 0
            font = pygame.font.SysFont(font, int(self.resolution/1.25))
            screen = pygame.display.set_mode([self.resolution,self.resolution])
            for _ in range(images_per_color): 
                for letter_index, (letter_color_name, letter_color) in enumerate(self.colors.items()):
                    for shape_index, (shape_color_name, shape_color), in enumerate(self.colors.items()):
                        if shape_color_name == letter_color_name:
                            continue
                        
                        currentDeg = 0
                        text = ALPHABET[random.randint(0,34)]
                        currentDeg += random.randint(-50, 65)
                        screen.fill((0,0,0)) # Fill screen with black
                        polygon_surface = self.generate_polygon(random.randint(3,5), shape_color)
                        rotated_polygon_surface = pygame.transform.rotate(polygon_surface, random.randint(0,360))
                        rotated_rect = rotated_polygon_surface.get_rect(center=(64,64))
                        self.screen.blit(rotated_polygon_surface, rotated_rect.topleft)

                        letter = font.render(text, True, letter_color)
                        letter = pygame.transform.rotate(letter, currentDeg)
                        textRect = letter.get_rect()
                        textRect.center = (self.resolution // 2, self.resolution // 2)
                        screen.blit(letter, textRect)
                        pygame.display.flip()
                        pygame.image.save(screen, f".{self.path}/dataset/data/{startNum}.jpg")
                        file.write(f"./data/{startNum}.jpg" + ", " + str(letter_index) +", " + str(shape_index) + "\n")
                        startNum += 1
        return startNum


    # def generate_data(self, font, numImages = 15, path=""):
    #     '''Generates data for the given font and number of images'''
    #     pygame.init()
    #     changeDeg = 360 // numImages
    #     currentDeg = 0
    #     numPhotos = 360 // changeDeg
    #     font = pygame.font.SysFont(font, self.resolution)
    #     screen = pygame.display.set_mode([self.resolution,self.resolution])
        
    #     for x in ALPHABET: 
    #         text = x
    #         for number in range(numPhotos):

    #             screen.fill((0,0,0)) # Fill screen with black
    #             letter = font.render(text, True, (255,255,255), (0,0,0))
    #             letter = pygame.transform.rotate(letter, currentDeg)
    #             currentDeg += changeDeg
    #             textRect = letter.get_rect()
    #             textRect.center = (self.resolution // 2, self.resolution // 2)
    #             screen.blit(letter, textRect)
    #             pygame.display.flip()
    #             pygame.image.save(screen, f"{path}/data/{text}/{number}.jpg")
    #     pygame.quit()
