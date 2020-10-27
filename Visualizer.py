import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import Biomes
import GenLayers

NUM_BIOMES = 64

hyperbiome_colors = np.array([
    [21, 128, 209],  # ocean 0
    [227, 217, 159],  # warm 1
    [163, 207, 114],  # medium
    [45, 102, 43],  # cold 3
    [209, 240, 238],  # ice 4
    [87, 207, 54],  # 5
    [87, 207, 54],  # 6
    [87, 207, 54],  # 7
    [87, 207, 54],  # 8
    [87, 207, 54],  # 9
    [87, 207, 54],  # 10
    [87, 207, 54],  # 11
    [87, 207, 54],  # 12
    [87, 207, 54],  # 13
    [224, 171, 245],  # mushroom_island 14
    [87, 207, 54],  # 15
    [87, 207, 54],  # 16
    [87, 207, 54],  # 17
    [87, 207, 54],  # 18
    [87, 207, 54],  # 19
    [87, 207, 54],  # 20
    [87, 207, 54],  # 21
    [87, 207, 54],  # 22
    [87, 207, 54],  # 23
    [21, 109, 209],  # deep_ocean 24
], dtype=float)

biome_colors = np.array([
    [21, 128, 209],  # ocean 0
    [129, 194, 101],  # plains 1
    [227, 217, 159],  # desert 2
    [184, 194, 178],  # extreme_hills 3
    [55, 125, 52],  # forest 4
    [45, 102, 43],  # taiga 5
    [78, 125, 25],  # swampland 6
    [23, 173, 232],  # river 7
    [148, 41, 15],  # hell 8
    [245, 245, 245],  # sky 9
    [181, 217, 245],  # frozen_ocean 10
    [181, 217, 245],  # frozen_river 11
    [209, 240, 238],  # ice_flats 12
    [235, 245, 244],  # ice_mountains 13
    [224, 171, 245],  # mushroom_island 14
    [213, 170, 230],  # mushroom_island_shore 15
    [227, 217, 159],  # beaches 16
    [227, 217, 159],  # desert_hills 17
    [71, 145, 68],  # forest_hills 18
    [45, 102, 43],  # taiga_hills 19
    [169, 199, 151],  # smaller_extreme_hills 20
    [56, 186, 0],  # jungle 21
    [86, 204, 35],  # jungle_hills 22
    [49, 138, 11],  # jungle_edge 23
    [21, 109, 209],  # deep_ocean 24
    [166, 165, 151],  # stone_beach 25
    [224, 220, 193],  # cold_beach 26
    [212, 235, 145],  # birch_forest 27
    [223, 240, 173],  # birch_forest_hills 28
    [59, 130, 5],  # roofed_forest 29
    [121, 140, 122],  # taiga_cold 30
    [145, 161, 145],  # taiga_cold_hills 31
    [56, 82, 42],  # redwood_taiga 32
    [74, 99, 61],  # redwood_taiga_hills 33
    [167, 199, 157],  # extreme_hills_with_trees 34
    [175, 181, 69],  # savanna 35
    [171, 170, 114],  # savanna_rock 36
    [115, 66, 44],  # mesa 37
    [107, 81, 70],  # mesa_rock 38
    [143, 114, 101],  # mesa_clear_rock 39
], dtype=float)

hyperbiome_colors /= 255
biome_colors /= 255
hb_cm = ListedColormap(hyperbiome_colors, N=NUM_BIOMES)
b_cm = ListedColormap(biome_colors, N=NUM_BIOMES)


def show_hyperbiome_map(arr, title="", file_name=None, extent=None):
    arr = arr % (1 << 8)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(arr), cmap=hb_cm, vmin=0, vmax=NUM_BIOMES-1, interpolation='nearest', extent=extent)
    plt.title(title)
    if file_name is not None:
        plt.savefig(file_name)
        plt.clf()
    else:
        plt.show()

def show_biome_map(arr, title="", file_name=None, extent=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(arr) % NUM_BIOMES, cmap=b_cm, vmin=0, vmax=NUM_BIOMES-1, interpolation='nearest', extent=extent)
    plt.title(title)
    if file_name is not None:
        plt.savefig(file_name)
        plt.clf()
    else:
        plt.show()

def show_gen_procedure(layer: GenLayers.GenLayer, size: int, is_hyper_biomes=False, save_gif=False):
    count = 0
    if layer.parent is not None:
        if isinstance(layer, GenLayers.Zoom):
            count = show_gen_procedure(layer.parent, size // 2, is_hyper_biomes=is_hyper_biomes, save_gif=save_gif)
        elif isinstance(layer, GenLayers.Biome):
            count = show_gen_procedure(layer.parent, size, is_hyper_biomes=True, save_gif=save_gif)
        else:
            count = show_gen_procedure(layer.parent, size, is_hyper_biomes=is_hyper_biomes, save_gif=save_gif)
    x = y = -size // 2
    width = height = size + 1
    image = layer.get_ints(x, y, width, height)
    file_name = f"gif/{count}.png" if save_gif else None
    if is_hyper_biomes:
        show_hyperbiome_map(image, title=str(type(layer)), file_name=file_name, extent=[x, x + width, y + height, y])
    else:
        show_biome_map(image, title=str(type(layer)), file_name=file_name, extent=[x, x+width, y+height, y])
    return count + 1

# Set up all the layers
island = GenLayers.Island(1)
fuzzy_zoom = GenLayers.FuzzyZoom(2000, island)
add_island1 = GenLayers.AddIsland(1, fuzzy_zoom)
zoom1 = GenLayers.Zoom(2001, add_island1)
add_island2 = GenLayers.AddIsland(2, zoom1)
add_island2 = GenLayers.AddIsland(50, add_island2)
add_island2 = GenLayers.AddIsland(70, add_island2)
remove_too_much_ocean = GenLayers.RemoveTooMuchOcean(2, add_island2)
add_snow = GenLayers.AddSnow(2, remove_too_much_ocean)
add_island3 = GenLayers.AddIsland(3, add_snow)
edge_cool_warm = GenLayers.EdgeCoolWarm(2, add_island3)
edge_heat_ice = GenLayers.EdgeHeatIce(2, edge_cool_warm)
edge_special = GenLayers.EdgeSpecial(3, edge_heat_ice)
zoom2 = GenLayers.Zoom(2002, edge_special)
zoom2 = GenLayers.Zoom(2003, zoom2)
add_island4 = GenLayers.AddIsland(4, zoom2)
mushroom = GenLayers.AddMushroomIsland(5, add_island4)
deep_ocean = GenLayers.DeepOcean(4, mushroom)
# Also, this is all taken from 1.12, so there are no fancy ocean biomes

biome = GenLayers.Biome(200, deep_ocean)
river_init = GenLayers.RiverInit(100, deep_ocean)
river_zoom1 = GenLayers.Zoom(1000, river_init)
river_zoom1 = GenLayers.Zoom(1001, river_zoom1)
zoom3 = GenLayers.Zoom(1000, biome)
zoom3 = GenLayers.Zoom(1001, zoom3)
biome_edge = GenLayers.BiomeEdge(1000, zoom3)
hills = GenLayers.Hills(1000, biome_edge, river_zoom1) # hills also should add mutated biomes, but I was too lazy to add those :\
river_zoom2 = GenLayers.Zoom(1000, river_zoom1)
river_zoom2 = GenLayers.Zoom(1001, river_zoom2)
river_zoom2 = GenLayers.Zoom(1002, river_zoom2)
river_zoom2 = GenLayers.Zoom(1003, river_zoom2)
# rare_biome - There should be a layer here that turns some plains into sunflower plains, but I was also too lazy add it.
# Also, what's up with having a layer to make sunflower plains? That's just mutated plains. Should that already be handled by hills? Mojang your game makes no sense.
zoom4 = GenLayers.Zoom(1000, hills)
add_island5 = GenLayers.AddIsland(3, zoom4)
zoom5 = GenLayers.Zoom(1001, add_island5)
shore = GenLayers.Shore(1000, zoom5)
zoom6 = GenLayers.Zoom(1002, shore)
zoom6 = GenLayers.Zoom(1003, zoom6)
smooth = GenLayers.Smooth(1000, zoom6)

river = GenLayers.River(1, river_zoom2)
river_smooth = GenLayers.Smooth(1000, river)

river_mix = GenLayers.RiverMix(100, smooth, river_smooth)
# There should be a layer here called voronoi_zoom, which does a fancy x4 zoom, but I'm lazy

def main():
    # EDIT THIS TO CHANGE THE WORLD SEED
    SEED = 6

    river_mix.init_world_gen_seed(SEED)

    # Use show_gen_procedure() to see the full generation process.
    # Enter the final layer and final size you want.
    # Set save_gif=True to save a sequence of pngs instead of showing all images.
    show_gen_procedure(river_mix, 2048, save_gif=False)

    # Show a single layer by using show_biome_map() or show_hyperbiome_map()
    # Pass in the result of GenLayer.get_ints(x, y, width, height)
    # Example:
    # show_biome_map(river_mix.get_ints(-256, -256, 512, 512))


if __name__ == "__main__":
    main()
