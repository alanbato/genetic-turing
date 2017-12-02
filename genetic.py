from collections import defaultdict, namedtuple
import random
from pprint import pprint
from copy import deepcopy
from math import floor

Input = namedtuple('Input', ['state', 'read'])
Result = namedtuple('Result', ['state', 'write', 'direction'])


def to_tape(data, blank):
    tape = [blank, blank, *list(data), blank, blank]
    return tape


class Individual:
    def __init__(self, rules=None, num_inputs=None, num_outputs=None):
        if rules is not None:
            self.rules = rules
        else:
            assert num_inputs and num_outputs
            self.rules = random.sample(range(num_outputs), num_inputs)
        self.num_inputs = num_inputs or len(self.rules)
        self.num_outputs = num_outputs or max(self.rules)

    def __sub__(self, other):
        diff_vector = []
        for rule, other_rule in zip(self.rules, other.rules):
            diff_vector.append(rule - other_rule)
        return diff_vector

    def __iter__(self):
        return (rule for rule in self.rules)

    def add_noise(self, diffs):
        return Individual(rules=[floor(original + diff)
                                 if floor(original + diff) > 0
                                 else 0
                                 for original, diff
                                 in zip(self.rules, diffs)])


class TransitionTable:
    def __init__(self):
        self._table = {}

    def __str__(self):
        return '{}'.format(self._table)

    def __repr__(self):
        return '{!r}'.format(self._table)

    def __iter__(self):
        return ([inp, out] for inp, out in sorted(self._table.items()))

    def add_rule(self, input_rule, result):
        self._table[input_rule] = result

    def evaluate(self, input_rule):
        try:
            return self._table[input_rule]
        except:
            return None

    def prune(self):
        out_states = [out.state for out in self._table.values()]
        to_prune = []
        for inp in self._table:
            if inp.state not in out_states and inp.state != 'q0':
                to_prune.append(inp)
        for i, rule in enumerate(to_prune):
            self._table.pop(rule)
        print('Rules pruned: {}'.format(i))


class Turing:
    def __init__(self, num_states, alphabet, symbols, transition, blank):
        self.states = ['q{}'.format(i) for i in range(num_states)]
        self.alphabet = alphabet
        self.symbols = symbols
        self.transition = transition
        self.initial_state = self.states[0]
        self.blank = blank
        self.final_states = [self.states[-1]]
        # Classical Optimization Encoding
        self.input_list = [Input(state, symbol)
                           for state in self.states
                           for symbol in self.alphabet]
        self.output_list = [Result(state, symbol, direction)
                            for state in self.states
                            for symbol in self.symbols
                            for direction in (-1, 0, 1)]
        self.num_inputs = len(self.input_list)
        self.num_outputs = len(self.output_list)
        # Differential Evolution
        self.dimension = (len(self.states) - 1) * len(self.symbols)

    def predict(self, input_str):
        state = self.initial_state
        head = 2
        tape = to_tape(input_str, self.blank)
        TTL = 1000
        while state not in self.final_states:
            if TTL == 0:
                return None
            if head >= len(tape):
                tape.append(self.blank)
            elif head < 0:
                tape.insert(0, self.blank)
                head = 0
            input_symbol = tape[head]
            result = self.transition.evaluate(Input(state, input_symbol))
            if result is None:
                break
            state = result.state
            tape[head] = result.write
            head += result.direction
            TTL -= 1
        return ''.join(tape)[2:-2]

    def print_tape_str(self, tape, head):
        tape_str = ' {}'.format(' '.join(tape))
        arrow_list = [' '] * len(tape_str)
        arrow_list[(head + 1) * 2 - 1] = '^'
        arrow_str = ''.join(arrow_list)
        print(tape_str)
        print(arrow_str)

    def ppredict(self, input_str):
        state = self.initial_state
        head = 2
        tape = to_tape(input_str, self.blank)
        TTL = 1000
        steps = 0
        print('Step 0')
        self.print_tape_str(tape, head)
        while state not in self.final_states:
            steps += 1
            if TTL == 0:
                return None
            input_symbol = tape[head]
            result = self.transition.evaluate(Input(state, input_symbol))
            if result is None:
                break
            state = result.state
            tape[head] = result.write
            head += result.direction
            if head >= len(tape):
                tape.append(self.blank)
            elif head < 0:
                tape.insert(0, self.blank)
                head = 0
            TTL -= 1
            print('Step {}'.format(steps))
            self.print_tape_str(tape, head)
        print('Total steps: {}'.format(steps))
        return ''.join(tape)[2:-2]

    def evaluate(self, input_str, output_str):
        predicted = self.predict(input_str)
        if predicted is None:
            return 0
        correct_counter = 0
        for predicted_value, real in zip(predicted, output_str):
            if predicted_value == real:
                correct_counter += 1
        # Error rate
        cost_value = (correct_counter / len(predicted))
        return cost_value

    def from_individual(self, individual):
        self.transition = TransitionTable()
        for input_idx, output_idx in enumerate(individual.rules):
            if output_idx >= self.num_outputs:
                output_idx = self.num_outputs - 1
            self.transition.add_rule(self.input_list[input_idx],
                                     self.output_list[output_idx])

    def print_rules(self):
        for inp, out in self.transition:
            print('({}, {}) -> ({}, {}, {})'.format(*inp, *out))

    def prune_rules(self):
        self.transition.prune()


class ClassicalOptimizer:
    def __init__(self, turing, NP=50, G=200, F=0.8, CR=0.7, V=20, CV=0):
        self.turing = turing
        self.NP = NP
        self.G = G
        self.F = F
        self.CR = CR
        self.V = V
        self.CV = CV
        self.input = None
        self.output = None
        self.train_data = None

    def generate_population(self):
        population = []
        for i in range(self.NP):
            population.append(Individual(num_inputs=self.turing.num_inputs,
                                         num_outputs=self.turing.num_outputs))
        return population

    def parse(self, description):
        return tuple(description.strip().split(' '))

    def parse_line(self, line):
        return [self.parse(line)]

    def parse_file(self, filename):
        data = []
        for line in open(filename):
            data.append(self.parse(line))
        return data

    def cost_fn(self, individual):
        self.turing.from_individual(individual)
        fitnesses = []
        for input_str, expected in self.train_data:
            fitnesses.append(self.turing.evaluate(input_str, expected))
        avg_fitness = sum(fitnesses) / len(fitnesses)
        return avg_fitness

    def evolve_generation(self, population):
        assert len(population) >= 4
        new_population = []
        for individual in population:
            surrogates = []
            for i in range(3):
                surrogate = random.choice(population)
                # If we get a dupe, pick again
                while (surrogate == individual or
                       surrogate in surrogates):
                    surrogate = random.choice(population)
                surrogates.append(surrogate)
            diff_vector = surrogates[0] - surrogates[1]
            weighted_vector = [diff * self.F for diff in diff_vector]
            noise_individual = surrogates[2].add_noise(weighted_vector)
            test_individual = Individual([test if random.random() < self.CR
                                          else original
                                          for original, test
                                          in zip(individual, noise_individual)
                                          ])
            survivor = max(individual, test_individual, key=self.cost_fn)
            new_population.append(survivor)
        return new_population

    def optimize(self, train_data):
        self.train_data = train_data
        population = self.generate_population()
        for i in range(1, self.G + 1):
            new_generation = self.evolve_generation(population)
            population = new_generation
            best_so_far = max(self.cost_fn(indv) for indv in population)
            best_found = best_so_far == 1.0
            if i % 10 == 0 or best_found:
                print('Generation: {}\tBest: {}'.format(i, best_so_far))
            if best_found:
                break
        best_individual = max(population, key=self.cost_fn)
        turing = deepcopy(self.turing)
        turing.from_individual(best_individual)
        return turing


if __name__ == '__main__':
    # Define Turing Machine
    num_states = 3
    symbols = ['0', '1', 'X', 'Y']
    blank = 'B'
    alphabet = ['0', '1', blank]
    transition = TransitionTable()
    turing = Turing(num_states, alphabet, symbols, transition, blank,)
    optimizer = ClassicalOptimizer(turing, NP=10 * turing.dimension)
    # Read one line
    # train_data = optimizer.parse_line('1001010 0110101')
    # Read a File
    train_data = optimizer.parse_file('input.txt')
    model = optimizer.optimize(train_data)
    for i, to_predict in enumerate(('1111#0000', '10#10', '01#01', '11#00')):
        print()
        print('Simulacion {}'.format(i))
        print()
        result = model.ppredict(to_predict)
        print('{} -> {}'.format(to_predict, result))
