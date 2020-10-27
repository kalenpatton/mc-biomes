import numpy as np
import Biomes


def convolute(x: np.int64, a: np.int64):
    return (x * (x * np.int64(6364136223846793005) + np.int64(1442695040888963407))) + a


class GenLayer:
    def __init__(self, base_seed):
        self.base_seed: np.int64 = np.int64(base_seed)
        self.base_seed = convolute(self.base_seed, base_seed)
        self.base_seed = convolute(self.base_seed, base_seed)
        self.base_seed = convolute(self.base_seed, base_seed)

        self.world_gen_seed: np.int64 = np.int64(0)
        self.parent: GenLayer = None
        self.chunk_seed: np.int64 = np.int64(0)

    def init_world_gen_seed(self, seed):
        self.world_gen_seed = np.int64(seed)

        if self.parent is not None:
            self.parent.init_world_gen_seed(seed)

        self.world_gen_seed = convolute(self.world_gen_seed, self.base_seed)
        self.world_gen_seed = convolute(self.world_gen_seed, self.base_seed)
        self.world_gen_seed = convolute(self.world_gen_seed, self.base_seed)

    def init_chunk_seed(self, x, z):
        x = np.int64(x)
        z = np.int64(z)
        self.chunk_seed = self.world_gen_seed
        self.chunk_seed = convolute(self.chunk_seed, x)
        self.chunk_seed = convolute(self.chunk_seed, z)
        self.chunk_seed = convolute(self.chunk_seed, x)
        self.chunk_seed = convolute(self.chunk_seed, z)

    def next_int(self, r):
        i: np.int32 = np.int32((self.chunk_seed >> 24) % np.int64(r))

        if i < 0:
            i += np.int32(r)

        self.chunk_seed = convolute(self.chunk_seed, self.world_gen_seed)
        return i

    def select_random(self, *choices):
        return choices[self.next_int(len(choices))]

    def biomes_equal_or_mesa_plateau(self, a, b):
        if a == b:
            return True
        if a == Biomes.MESA_ROCK or a == Biomes.MESA_CLEAR_ROCK:
            return b == Biomes.MESA_ROCK or b == Biomes.MESA_CLEAR_ROCK
        return Biomes.get_biome_class(a) == Biomes.get_biome_class(b)

    def is_oceanic(self, a):
        return a == Biomes.OCEAN or a == Biomes.DEEP_OCEAN or a == Biomes.FROZEN_OCEAN

    def get_ints(self, x, y, width, height):
        raise NotImplemented


class Island(GenLayer):
    def __init__(self, base_seed):
        super().__init__(base_seed)

    def get_ints(self, x, y, width, height):
        new_arr = np.ndarray((width, height), dtype=int)
        for i in range(width):
            for j in range(height):
                self.init_chunk_seed(x + i, y + j)
                new_arr[i][j] = 1 if self.next_int(10) == 0 else 0

        if -width < x <= 0 and -height < y <= 0:
            new_arr[-x][-y] = 1

        return new_arr


class Zoom(GenLayer):
    def __init__(self, base_seed, parent):
        super().__init__(base_seed)
        self.parent: GenLayer = parent

    def mode_or_random(self, a, b, c, d):
        if b == c and c == d:
            return b
        elif a == b and a == c:
            return a
        elif a == b and a == d:
            return a
        elif a == c and a == d:
            return a
        elif a == b and c != d:
            return a
        elif a == c and b != d:
            return a
        elif a == d and b != c:
            return a
        elif b == c and a != d:
            return b
        elif b == d and a != c:
            return b
        return c if c == d and a != b else self.select_random(a, b, c, d)

    def get_ints(self, x, y, width, height):
        px = x >> 1
        py = y >> 1
        pwidth = (width >> 1) + 2
        pheight = (height >> 1) + 2
        p_arr = self.parent.get_ints(px, py, pwidth, pheight)
        iwidth = (pwidth - 1) << 1
        iheight = (pheight - 1) << 1

        i_arr = np.ndarray((iwidth, iheight), dtype=int)
        for j in range(pheight - 1):
            for i in range(pwidth - 1):
                self.init_chunk_seed((px + i) * 2, (py + j) * 2)
                i_arr[2 * i][2 * j] = p_arr[i][j]
                i_arr[2 * i][2 * j + 1] = self.select_random(p_arr[i][j], p_arr[i][j + 1])
                i_arr[2 * i + 1][2 * j] = self.select_random(p_arr[i][j], p_arr[i + 1][j])
                i_arr[2 * i + 1][2 * j + 1] = self.mode_or_random(p_arr[i][j], p_arr[i + 1][j], p_arr[i][j + 1],
                                                                  p_arr[i + 1][j + 1])

        startx = x & 1
        starty = y & 1
        new_arr = i_arr[startx:(startx + width), starty:(starty + height)]

        return new_arr


class FuzzyZoom(Zoom):
    def mode_or_random(self, a, b, c, d):
        return self.select_random(a, b, c, d)


class AddIsland(GenLayer):
    def __init__(self, base_seed, parent):
        super().__init__(base_seed)
        self.parent: GenLayer = parent

    def get_ints(self, x, y, width, height):
        new_arr = np.ndarray((width, height), dtype=int)
        px = x - 1
        py = y - 1
        pwidth = width + 2
        pheight = height + 2
        p_arr = self.parent.get_ints(px, py, pwidth, pheight)

        for i in range(width):
            for j in range(height):
                self.init_chunk_seed(x + i, y + j)
                a = p_arr[i][j]
                b = p_arr[i + 2][j]
                c = p_arr[i][j + 2]
                d = p_arr[i + 2][j + 2]
                k = p_arr[i + 1][j + 1]

                if k != 0 or (a == 0 and b == 0 and c == 0 and d == 0):
                    if k > 0 and (a == 0 or b == 0 or c == 0 or d == 0):
                        if self.next_int(5) == 0:
                            if k == 4:
                                new_arr[i][j] = 4
                            else:
                                new_arr[i][j] = 0
                        else:
                            new_arr[i][j] = k
                    else:
                        new_arr[i][j] = k
                else:
                    l2 = 1
                    n = 1
                    if a != 0:
                        if self.next_int(l2) == 0:
                            n = a
                        l2 += 1
                    if b != 0:
                        if self.next_int(l2) == 0:
                            n = b
                        l2 += 1
                    if c != 0:
                        if self.next_int(l2) == 0:
                            n = c
                        l2 += 1
                    if d != 0:
                        if self.next_int(l2) == 0:
                            n = d
                        l2 += 1

                    if self.next_int(3) == 0:
                        new_arr[i][j] = n
                    elif n == 4:
                        new_arr[i][j] = 4
                    else:
                        new_arr[i][j] = 0

        return new_arr


class RemoveTooMuchOcean(GenLayer):
    def __init__(self, base_seed, parent):
        super().__init__(base_seed)
        self.parent: GenLayer = parent

    def get_ints(self, x, y, width, height):
        new_arr = np.ndarray((width, height), dtype=int)
        px = x - 1
        py = y - 1
        pwidth = width + 2
        pheight = height + 2
        p_arr = self.parent.get_ints(px, py, pwidth, pheight)

        for j in range(height):
            for i in range(width):
                self.init_chunk_seed(x + i, y + j)
                a = p_arr[i + 1][j]
                b = p_arr[i + 2][j + 1]
                c = p_arr[i][j + 1]
                d = p_arr[i + 1][j + 2]
                k = p_arr[i + 1][j + 1]

                new_arr[i][j] = k
                if k == 0 and a == 0 and b == 0 and c == 0 and d == 0 and self.next_int(2) == 0:
                    new_arr[i][j] = 1

        return new_arr


class AddSnow(GenLayer):
    def __init__(self, base_seed, parent):
        super().__init__(base_seed)
        self.parent: GenLayer = parent

    def get_ints(self, x, y, width, height):
        new_arr = np.ndarray((width, height), dtype=int)
        p_arr = self.parent.get_ints(x, y, width, height)

        for j in range(height):
            for i in range(width):
                self.init_chunk_seed(x + i, y + j)
                k = p_arr[i][j]
                if k == 0:
                    new_arr[i][j] = 0
                else:
                    r = self.next_int(6)
                    if r == 0:
                        r = 4
                    elif r <= 1:
                        r = 3
                    else:
                        r = 1
                    new_arr[i][j] = r

        return new_arr


class EdgeCoolWarm(GenLayer):
    def __init__(self, base_seed, parent):
        super().__init__(base_seed)
        self.parent: GenLayer = parent

    def get_ints(self, x, y, width, height):
        new_arr = np.ndarray((width, height), dtype=int)
        px = x - 1
        py = y - 1
        pwidth = width + 2
        pheight = height + 2
        p_arr = self.parent.get_ints(px, py, pwidth, pheight)

        for j in range(height):
            for i in range(width):
                k = p_arr[i + 1][j + 1]

                if k == 1:
                    a = p_arr[i + 1][j]
                    b = p_arr[i + 2][j + 1]
                    c = p_arr[i][j + 1]
                    d = p_arr[i + 1][j + 2]
                    flag1 = a == 3 or b == 3 or c == 3 or d == 3
                    flag2 = a == 4 or b == 4 or c == 4 or d == 4

                    if flag1 or flag2:
                        k = 2
                new_arr[i][j] = k

        return new_arr


class EdgeHeatIce(GenLayer):
    def __init__(self, base_seed, parent):
        super().__init__(base_seed)
        self.parent: GenLayer = parent

    def get_ints(self, x, y, width, height):
        new_arr = np.ndarray((width, height), dtype=int)
        px = x - 1
        py = y - 1
        pwidth = width + 2
        pheight = height + 2
        p_arr = self.parent.get_ints(px, py, pwidth, pheight)

        for j in range(height):
            for i in range(width):
                k = p_arr[i + 1][j + 1]

                if k == 4:
                    a = p_arr[i + 1][j]
                    b = p_arr[i + 2][j + 1]
                    c = p_arr[i][j + 1]
                    d = p_arr[i + 1][j + 2]
                    flag1 = a == 2 or b == 2 or c == 2 or d == 2
                    flag2 = a == 1 or b == 1 or c == 1 or d == 1

                    if flag1 or flag2:
                        k = 3
                new_arr[i][j] = k

        return new_arr


class EdgeSpecial(GenLayer):
    def __init__(self, base_seed, parent):
        super().__init__(base_seed)
        self.parent: GenLayer = parent

    def get_ints(self, x, y, width, height):
        new_arr = np.ndarray((width, height), dtype=int)
        p_arr = self.parent.get_ints(x, y, width, height)

        for j in range(height):
            for i in range(width):
                self.init_chunk_seed(x + i, y + j)
                k = p_arr[i][j]
                if k != 0 and self.next_int(13) == 0:
                    k = int(k) | (((1 + self.next_int(15)) << 8) & 3840)
                new_arr[i][j] = k

        return new_arr


class AddMushroomIsland(GenLayer):
    def __init__(self, base_seed, parent):
        super().__init__(base_seed)
        self.parent: GenLayer = parent

    def get_ints(self, x, y, width, height):
        new_arr = np.ndarray((width, height), dtype=int)
        px = x - 1
        py = y - 1
        pwidth = width + 2
        pheight = height + 2
        p_arr = self.parent.get_ints(px, py, pwidth, pheight)

        for j in range(height):
            for i in range(width):
                self.init_chunk_seed(x + i, y + j)
                a = p_arr[i][j]
                b = p_arr[i + 2][j]
                c = p_arr[i][j + 2]
                d = p_arr[i + 2][j + 2]
                k = p_arr[i + 1][j + 1]

                if k == 0 and a == 0 and b == 0 and c == 0 and d == 0 and self.next_int(100) == 0:
                    new_arr[i][j] = Biomes.MUSHROOM_ISLAND
                else:
                    new_arr[i][j] = k

        return new_arr


class DeepOcean(GenLayer):
    def __init__(self, base_seed, parent):
        super().__init__(base_seed)
        self.parent: GenLayer = parent

    def get_ints(self, x, y, width, height):
        new_arr = np.ndarray((width, height), dtype=int)
        px = x - 1
        py = y - 1
        pwidth = width + 2
        pheight = height + 2
        p_arr = self.parent.get_ints(px, py, pwidth, pheight)

        for j in range(height):
            for i in range(width):
                a = p_arr[i + 1][j]
                b = p_arr[i + 2][j + 1]
                c = p_arr[i][j + 1]
                d = p_arr[i + 1][j + 2]
                k = p_arr[i + 1][j + 1]

                if k == 0 and a == 0 and b == 0 and c == 0 and d == 0:
                    new_arr[i][j] = Biomes.DEEP_OCEAN
                else:
                    new_arr[i][j] = k

        return new_arr


class Biome(GenLayer):
    def __init__(self, base_seed, parent):
        super().__init__(base_seed)
        self.parent: GenLayer = parent

    def get_ints(self, x, y, width, height):
        new_arr = np.ndarray((width, height), dtype=int)
        p_arr = self.parent.get_ints(x, y, width, height)

        for j in range(height):
            for i in range(width):
                self.init_chunk_seed(x + i, y + j)
                k = p_arr[i][j]
                r = (int(k) & 3840) >> 8
                k = int(k) & -3841

                if k == 0 or k == Biomes.DEEP_OCEAN or k == Biomes.FROZEN_OCEAN or k == Biomes.MUSHROOM_ISLAND:
                    new_arr[i][j] = k
                elif k == 1:
                    if r > 0:
                        if self.next_int(3) == 0:
                            new_arr[i][j] = Biomes.MESA_CLEAR_ROCK
                        else:
                            new_arr[i][j] = Biomes.MESA_ROCK
                    else:
                        new_arr[i][j] = self.select_random(*Biomes.WARM_BIOMES)
                elif k == 2:
                    if r > 0:
                        new_arr[i][j] = Biomes.JUNGLE
                    else:
                        new_arr[i][j] = self.select_random(*Biomes.MEDIUM_BIOMES)
                elif k == 3:
                    if r > 0:
                        new_arr[i][j] = Biomes.REDWOOD_TAIGA
                    else:
                        new_arr[i][j] = self.select_random(*Biomes.COLD_BIOMES)
                elif k == 4:
                    new_arr[i][j] = self.select_random(*Biomes.ICE_BIOMES)
                else:
                    new_arr[i][j] = Biomes.MUSHROOM_ISLAND

        return new_arr


class BiomeEdge(GenLayer):
    def __init__(self, base_seed, parent):
        super().__init__(base_seed)
        self.parent: GenLayer = parent

    def can_be_neighbors(self, a, b):
        if self.biomes_equal_or_mesa_plateau(a, b):
            return True
        temp_a = Biomes.get_temp_category(a)
        temp_b = Biomes.get_temp_category(b)
        if temp_a == temp_b or temp_a == Biomes.TEMP_MEDIUM or temp_b == Biomes.TEMP_MEDIUM:
            return True
        return False

    def replace_biome_edge(self, p_arr, new_arr, i, j, k, to_replace, replace_with):
        if k != to_replace:
            return False
        else:
            a = p_arr[i + 1][j]
            b = p_arr[i + 2][j + 1]
            c = p_arr[i][j + 1]
            d = p_arr[i + 1][j + 2]

            if self.biomes_equal_or_mesa_plateau(a, k) and self.biomes_equal_or_mesa_plateau(b, k) \
                    and self.biomes_equal_or_mesa_plateau(c, k) and self.biomes_equal_or_mesa_plateau(d, k):
                new_arr[i][j] = int(to_replace)
            else:
                new_arr[i][j] = int(replace_with)
            return True

    def replace_biome_edge_if_necessary(self, p_arr, new_arr, i, j, k, to_replace, replace_with):
        if not self.biomes_equal_or_mesa_plateau(k, to_replace):
            return False
        else:
            a = p_arr[i + 1][j]
            b = p_arr[i + 2][j + 1]
            c = p_arr[i][j + 1]
            d = p_arr[i + 1][j + 2]

            if self.can_be_neighbors(a, k) and self.can_be_neighbors(b, k) \
                    and self.can_be_neighbors(c, k) and self.can_be_neighbors(d, k):
                new_arr[i][j] = to_replace
            else:
                new_arr[i][j] = replace_with
            return True

    def get_ints(self, x, y, width, height):
        new_arr = np.ndarray((width, height), dtype=int)
        px = x - 1
        py = y - 1
        pwidth = width + 2
        pheight = height + 2
        p_arr = self.parent.get_ints(px, py, pwidth, pheight)

        for j in range(height):
            for i in range(width):
                self.init_chunk_seed(x + i, y + j)
                k = p_arr[i + 1][j + 1]

                if not self.replace_biome_edge_if_necessary(p_arr, new_arr, i, j, k, Biomes.EXTREME_HILLS, Biomes.SMALLER_EXTREME_HILLS) \
                        and not self.replace_biome_edge(p_arr, new_arr, i, j, k, Biomes.MESA_ROCK, Biomes.MESA) \
                        and not self.replace_biome_edge(p_arr, new_arr, i, j, k, Biomes.MESA_CLEAR_ROCK, Biomes.MESA) \
                        and not self.replace_biome_edge(p_arr, new_arr, i, j, k, Biomes.REDWOOD_TAIGA, Biomes.TAIGA):

                    a = p_arr[i + 1][j]
                    b = p_arr[i + 2][j + 1]
                    c = p_arr[i][j + 1]
                    d = p_arr[i + 1][j + 2]

                    if k == Biomes.DESERT:
                        if a != Biomes.ICE_PLAINS and b != Biomes.ICE_PLAINS and c != Biomes.ICE_PLAINS and d != Biomes.ICE_PLAINS:
                            new_arr[i][j] = k
                        else:
                            new_arr[i][j] = Biomes.EXTREME_HILLS_WITH_TREES

                    elif k == Biomes.SWAMPLAND:
                        if a != Biomes.DESERT and b != Biomes.DESERT and c != Biomes.DESERT and d != Biomes.DESERT \
                                and a != Biomes.TAIGA_COLD and b != Biomes.TAIGA_COLD and c != Biomes.TAIGA_COLD and d != Biomes.TAIGA_COLD:
                            if a != Biomes.JUNGLE and b != Biomes.JUNGLE and c != Biomes.JUNGLE and d != Biomes.JUNGLE:
                                new_arr[i][j] = k
                            else:
                                new_arr[i][j] = Biomes.JUNGLE_EDGE
                        else:
                            new_arr[i][j] = Biomes.PLAINS
                    else:
                        new_arr[i][j] = k
        return new_arr


class RiverInit(GenLayer):
    def __init__(self, base_seed, parent):
        super().__init__(base_seed)
        self.parent: GenLayer = parent

    def get_ints(self, x, y, width, height):
        new_arr = np.ndarray((width, height), dtype=int)
        p_arr = self.parent.get_ints(x, y, width, height)

        for j in range(height):
            for i in range(width):
                self.init_chunk_seed(x + i, y + j)
                if p_arr[i][j] > 0:
                    new_arr[i][j] = self.next_int(299999) + 2
                else:
                    new_arr[i][j] = 0

        return new_arr


class River(GenLayer):
    def __init__(self, base_seed, parent):
        super().__init__(base_seed)
        self.parent: GenLayer = parent

    def river_filter(self, k):
        if k >= 2:
            return 2 + (k & 1)
        else:
            return k

    def get_ints(self, x, y, width, height):
        new_arr = np.ndarray((width, height), dtype=int)
        px = x - 1
        py = y - 1
        pwidth = width + 2
        pheight = height + 2
        p_arr = self.parent.get_ints(px, py, pwidth, pheight)

        for j in range(height):
            for i in range(width):
                a = self.river_filter(p_arr[i + 1][j])
                b = self.river_filter(p_arr[i + 2][j + 1])
                c = self.river_filter(p_arr[i][j + 1])
                d = self.river_filter(p_arr[i + 1][j + 2])
                k = self.river_filter(p_arr[i + 1][j + 1])

                if k == a and k == b and k == c and k == d:
                    new_arr[i][j] = -1
                else:
                    new_arr[i][j] = Biomes.RIVER

        return new_arr


class Hills(GenLayer):
    def __init__(self, base_seed, parent, river_layer):
        super().__init__(base_seed)
        self.parent: GenLayer = parent
        self.river_layer: GenLayer = river_layer

    def get_ints(self, x, y, width, height):
        new_arr = np.ndarray((width, height), dtype=int)
        px = x - 1
        py = y - 1
        pwidth = width + 2
        pheight = height + 2
        p_arr = self.parent.get_ints(px, py, pwidth, pheight)
        r_arr = self.river_layer.get_ints(x, y, width, height)

        for j in range(height):
            for i in range(width):
                self.init_chunk_seed(x + i, y + j)
                k = p_arr[i + 1][j + 1]
                l = r_arr[i][j]
                mutate_hills = l >= 2 and (l - 2) % 29 == 0


                if k != 0 and l >= 2 and (l-2) % 29 == 1 and not Biomes.is_mutation(k):
                    # Mutate biome
                    mutated = Biomes.get_mutation_for_biome(k)
                    new_arr[i][j] = mutated if mutated is not None else k
                elif self.next_int(3) != 0 and not mutate_hills:
                    new_arr[i][j] = k
                else:
                    n = k
                    if k == Biomes.DESERT:
                        n = Biomes.DESERT_HILLS
                    elif k == Biomes.FOREST:
                        n = Biomes.FOREST_HILLS
                    elif k == Biomes.BIRCH_FOREST:
                        n = Biomes.BIRCH_FOREST_HILLS
                    elif k == Biomes.ROOFED_FOREST:
                        n = Biomes.PLAINS
                    elif k == Biomes.TAIGA:
                        n = Biomes.TAIGA_HILLS
                    elif k == Biomes.REDWOOD_TAIGA:
                        n = Biomes.REDWOOD_TAIGA_HILLS
                    elif k == Biomes.TAIGA_COLD:
                        n = Biomes.TAIGA_COLD_HILLS
                    elif k == Biomes.PLAINS:
                        if self.next_int(3) == 0:
                            n = Biomes.FOREST_HILLS
                        else:
                            n = Biomes.FOREST
                    elif k == Biomes.ICE_PLAINS:
                        n = Biomes.ICE_MOUNTAINS
                    elif k == Biomes.JUNGLE:
                        n = Biomes.JUNGLE_HILLS
                    elif k == Biomes.OCEAN:
                        n = Biomes.DEEP_OCEAN
                    elif k == Biomes.EXTREME_HILLS:
                        n = Biomes.EXTREME_HILLS_WITH_TREES
                    elif k == Biomes.SAVANNA:
                        n = Biomes.SAVANNA_ROCK
                    elif self.biomes_equal_or_mesa_plateau(k, Biomes.MESA_ROCK):
                        n = Biomes.MESA
                    elif k == Biomes.DEEP_OCEAN and self.next_int(3) == 0:
                        c = self.next_int(2)
                        if c == 0:
                            n = Biomes.PLAINS
                        else:
                            n = Biomes.FOREST

                    if mutate_hills and n != k:
                        mutated = Biomes.get_mutation_for_biome(n)
                        n = mutated if mutated is not None else k
                    if n == k:
                        new_arr[i][j] = k
                    else:
                        a = p_arr[i + 1][j]
                        b = p_arr[i + 2][j + 1]
                        c = p_arr[i][j + 1]
                        d = p_arr[i + 1][j + 2]

                        num_same_neighbors = 0
                        if self.biomes_equal_or_mesa_plateau(a, k):
                            num_same_neighbors += 1
                        if self.biomes_equal_or_mesa_plateau(b, k):
                            num_same_neighbors += 1
                        if self.biomes_equal_or_mesa_plateau(c, k):
                            num_same_neighbors += 1
                        if self.biomes_equal_or_mesa_plateau(d, k):
                            num_same_neighbors += 1

                        if num_same_neighbors >= 3:
                            new_arr[i][j] = n
                        else:
                            new_arr[i][j] = k

        return new_arr


class Smooth(GenLayer):
    def __init__(self, base_seed, parent):
        super().__init__(base_seed)
        self.parent: GenLayer = parent

    def get_ints(self, x, y, width, height):
        new_arr = np.ndarray((width, height), dtype=int)
        px = x - 1
        py = y - 1
        pwidth = width + 2
        pheight = height + 2
        p_arr = self.parent.get_ints(px, py, pwidth, pheight)

        for j in range(height):
            for i in range(width):
                a = p_arr[i][j + 1]
                b = p_arr[i + 2][j + 1]
                c = p_arr[i + 1][j]
                d = p_arr[i + 1][j + 2]
                k = p_arr[i + 1][j + 1]

                if a == b and c == d:
                    self.init_chunk_seed(x + i, y + j)
                    if self.next_int(2) == 0:
                        k = a
                    else:
                        k = c
                else:
                    if a == b:
                        k = a
                    if c == d:
                        k = c
                new_arr[i][j] = k

        return new_arr


class RiverMix(GenLayer):
    def __init__(self, base_seed, parent, river_layer):
        super().__init__(base_seed)
        self.parent: GenLayer = parent
        self.river_layer: GenLayer = river_layer

    def init_world_gen_seed(self, seed):
        super().init_world_gen_seed(seed)
        if self.river_layer is not None:
            self.river_layer.init_world_gen_seed(seed)

    def get_ints(self, x, y, width, height):
        new_arr = np.ndarray((width, height), dtype=int)
        p_arr = self.parent.get_ints(x, y, width, height)
        r_arr = self.river_layer.get_ints(x, y, width, height)

        for j in range(height):
            for i in range(width):
                if p_arr[i][j] != Biomes.OCEAN and p_arr[i][j] != Biomes.DEEP_OCEAN:
                    if r_arr[i][j] == Biomes.RIVER:
                        if p_arr[i][j] == Biomes.ICE_PLAINS:
                            new_arr[i][j] = Biomes.FROZEN_RIVER
                        elif p_arr[i][j] != Biomes.MUSHROOM_ISLAND and p_arr[i][j] != Biomes.MUSHROOM_ISLAND_SHORE:
                            new_arr[i][j] = int(r_arr[i][j]) & 255
                        else:
                            new_arr[i][j] = Biomes.MUSHROOM_ISLAND_SHORE
                    else:
                        new_arr[i][j] = p_arr[i][j]
                else:
                    new_arr[i][j] = p_arr[i][j]
        return new_arr


class Shore(GenLayer):
    def __init__(self, base_seed, parent):
        super().__init__(base_seed)
        self.parent: GenLayer = parent

    def is_mesa(self, k):
        return k == Biomes.MESA or k == Biomes.MESA_ROCK or k == Biomes.MESA_CLEAR_ROCK

    def is_jungle_compatible(self, k):
        return k == Biomes.JUNGLE_EDGE or k == Biomes.JUNGLE or k == Biomes.JUNGLE_HILLS or k == Biomes.FOREST or k == Biomes.TAIGA or self.is_oceanic(k)

    def get_ints(self, x, y, width, height):
        new_arr = np.ndarray((width, height), dtype=int)
        px = x - 1
        py = y - 1
        pwidth = width + 2
        pheight = height + 2
        p_arr = self.parent.get_ints(px, py, pwidth, pheight)

        for j in range(height):
            for i in range(width):
                self.init_chunk_seed(x + i, y + j)
                a = p_arr[i][j + 1]
                b = p_arr[i + 2][j + 1]
                c = p_arr[i + 1][j]
                d = p_arr[i + 1][j + 2]
                k = p_arr[i + 1][j + 1]

                if k == Biomes.MUSHROOM_ISLAND:
                    if a != Biomes.OCEAN and b != Biomes.OCEAN and c != Biomes.OCEAN and d != Biomes.OCEAN:
                        new_arr[i][j] = k
                    else:
                        new_arr[i][j] = Biomes.MUSHROOM_ISLAND_SHORE
                elif k == Biomes.JUNGLE or k == Biomes.JUNGLE_HILLS or k == Biomes.JUNGLE_EDGE:
                    if self.is_jungle_compatible(a) and self.is_jungle_compatible(b) and self.is_jungle_compatible(c) and self.is_jungle_compatible(d):
                        if self.is_oceanic(a) or self.is_oceanic(b) or self.is_oceanic(c) or self.is_oceanic(d):
                            new_arr[i][j] = Biomes.BEACHES
                        else:
                            new_arr[i][j] = k
                    else:
                        new_arr[i][j] = Biomes.JUNGLE_EDGE
                elif k == Biomes.EXTREME_HILLS or k == Biomes.EXTREME_HILLS_WITH_TREES or k == Biomes.SMALLER_EXTREME_HILLS:
                    if self.is_oceanic(a) or self.is_oceanic(b) or self.is_oceanic(c) or self.is_oceanic(d):
                        new_arr[i][j] = Biomes.STONE_BEACH
                    else:
                        new_arr[i][j] = k
                elif Biomes.is_snowy(k):
                    if not self.is_oceanic(k) and (self.is_oceanic(a) or self.is_oceanic(b) or self.is_oceanic(c) or self.is_oceanic(d)):
                        new_arr[i][j] = Biomes.COLD_BEACH
                    else:
                        new_arr[i][j] = k
                elif k == Biomes.MESA or k == Biomes.MESA_ROCK:
                    if (not self.is_oceanic(a) and not self.is_oceanic(b) and not self.is_oceanic(c) and not self.is_oceanic(d)) \
                            and (not self.is_mesa(a) or not self.is_mesa(b) or not self.is_mesa(c) or not self.is_mesa(d)):
                        new_arr[i][j] = Biomes.DESERT
                    else:
                        new_arr[i][j] = k
                elif k != Biomes.OCEAN and k != Biomes.DEEP_OCEAN and k != Biomes.RIVER and k != Biomes.SWAMPLAND:
                    if self.is_oceanic(a) or self.is_oceanic(b) or self.is_oceanic(c) or self.is_oceanic(d):
                        new_arr[i][j] = Biomes.BEACHES
                    else:
                        new_arr[i][j] = k
                else:
                    new_arr[i][j] = k

        return new_arr