import torch
import torch.autograd as autograd
import random
import numpy as np
from tensorboardX import SummaryWriter

import time
import os

class SimpleOptimizer(torch.optim.Optimizer):
    def __init__(self, entity, lr=0.01):
        defaults = dict(lr=lr)
        self.entity = entity
        self.lr = lr

        #super(SimpleOptimizer, self).__init__([x.base_probability for x in entity.actions], defaults)

    def __setstate__(self, state):
        super(SimpleOptimizer, self).__setstate__(state)

    def zero_grad(self):
        """Clears the gradients of all optimized :class:`Variable` s."""
        for p in [x.base_probability for x in self.entity.actions]:
            if p.grad is not None:
                p.grad.detach_()

                p.grad.zero_()

    def step(self):
        for i, action in enumerate(self.entity.actions):
            p = action.base_probability
            #print(p.grad)
            if p.grad is None:
                continue

            d_p = torch.sign(p.grad.clone())

            #action.base_probability.data.add_(-self.param_groups[0]['lr'], d_p)

            #action.base_probability = autograd.Variable(torch.FloatTensor(action.base_probability.data -self.lr * d_p), requires_grad=True)
            #print(d_p)
            action.base_probability = action.base_probability - self.lr * d_p

            #if action.base_probability.grad is not None:
            #    action.base_probability.grad.backward()
            #p.data = p.data - self.lr * d_p.data
            #update = group['lr'] * d_p
            #group['params'][i] = (p - autograd.Variable(update)).detach()
            #p = p.detach()

class Zone:
    def __init__(self, name, traits, unique_entities=[], variables=[], subzones=[]):
        self.name = name

        self.traits = {}

        for trait in traits:
            self.traits[trait.name] = EntityTrait(trait.name, trait.intensity.data, trait.update_action)

        self._variables = {}
        for variable_name, variable_value in variables:
            self._variables[variable_name] = autograd.Variable(torch.FloatTensor(variable_value))

        self.subzones = subzones
        self.entities = list(unique_entities)

        self.saved_variables = None

    def add_possible_entities(self, entity_templates):
        for entity_template in entity_templates:
            if entity_template.check(self):
                self.entities.append(entity_template.constructor(self))

        for subzone in self.subzones:
            subzone.add_possible_entities(entity_templates)

    def has_subzones(self):
        return self.subzones != []

    def get_variable(self, name):
        if self.has_subzones():
            total = 0
            for subzone in self.subzones:
                total = total + subzone.get_variable(name)
            return total
        else:
            return self._variables[name]

    def set_variable(self, name, value):
        if self.has_subzones():
            raise Exception('Cannot directly set the value of a variable of a zone with subzones. '
                            'Use recursive_set_variable.')
        else:
            self._variables[name] = value

    def recursive_set_variable(self, name, value_function):
        if self.has_subzones():
            for subzone in self.subzones:
                subzone.recursive_set_variable(name, value_function)
        else:
            self._variables[name] = value_function(self._variables[name])

    def update_traits(self, time_coefficient=1):
        for name, trait in self.traits.items():
            trait.update(time_coefficient)

        if self.has_subzones():
            for subzone in self.subzones:
                subzone.update_traits(time_coefficient)

    def reset(self):
        if self.has_subzones():
            for subzone in self.subzones:
                subzone.reset()
        else:
            for variable_name, variable in self._variables.items():
                self._variables[variable_name] = autograd.Variable(torch.FloatTensor(variable.data.clone()))

            for trait_name, trait in self.traits.items():
                self.traits[trait_name].intensity = autograd.Variable(torch.FloatTensor(trait.intensity.data.clone()))

    def save_state(self):
        if self.has_subzones():
            for subzone in self.subzones:
                subzone.save_state()
        else:
            self.saved_variables = None

            for variable_name, variable in self._variables.items():
                self.saved_variables[variable_name] = variable.data.clone()

    def restore_state(self):
        if self.has_subzones():
            for subzone in self.subzones:
                subzone.restore_state()
        else:
            for variable_name, variable_value in self._variables.items():
                self._variables[variable_name].data = torch.FloatTensor([variable_value])
            self.saved_variables = None
            
    def get_all_entities(self):
        entities = list(self.entities)

        for subzone in self.subzones:
            entities += subzone.get_all_entities()

        return entities


class EntityTrait:
    def __init__(self, name, intensity=1.0, update_action=None):
        self.name = name

        if not isinstance(intensity, torch.Tensor):
            intensity = torch.FloatTensor([intensity])

        self.intensity = autograd.Variable(intensity)
        self.update_action = update_action

    def update(self, entity, time_coefficient=1):
        if self.update_action != None:
            self.update_action(entity, self.intensity * time_coefficient)

def check_traits(entity, requirements=[], blacklist=[]):
    entity_trait_names = [name for name, _ in entity.traits.items()]

    for requirement in requirements:
        if requirement not in entity_trait_names:
            return False

    for blacklist in blacklist:
        if blacklist in entity_trait_names:
            return False

    return True


class Template:
    def __init__(self, constructor):
        self.constructor = constructor

class EntityEventTemplate(Template):
    def __init__(self, constructor, requirements=[], blacklist=[], extra_condition=None):
        super().__init__(constructor)
        self.requirements = requirements
        self.blacklist = blacklist
        self.extra_condition = extra_condition

    def check(self, entity):
        if not check_traits(entity, self.requirements, self.blacklist):
            return False

        if self.extra_condition == None:
           return True
        else:
            return self.extra_condition(entity)

class EntityEvent(EntityTrait):
    def __init__(self, name, intensity = 1.0, stop_condition=None, update_action=None):
        super().__init__(name, intensity, update_action)
        self.stop_condition = stop_condition

class EntityActionTemplate(Template):
    def __init__(self, constructor):
        super().__init__(constructor)

class EntityAction:
    def __init__(self, name, flavour_text, approaches, initial_base_probability, costs, action, entity=None, target=None, action_type='Standard', entity_requirements=[], entity_blacklist=[], target_requirements=[], target_blacklist=[], extra_condition=None):
        self.name = name
        self.flavour_text = flavour_text.format(Entity=entity.name, Target=(None if (target == None) else target.name))
        self.approaches = approaches
        self.base_probability = autograd.Variable(torch.FloatTensor([initial_base_probability]), requires_grad=True)
        self.costs = costs
        self.action = action
        self.entity = entity
        self.target = target
        self.action_type = action_type
        self.entity_requirements = entity_requirements
        self.entity_blacklist = entity_blacklist
        self.target_requirements = target_requirements
        self.target_blacklist = target_blacklist
        self.extra_condition = extra_condition

        self.saved_probability = None

    def valid(self):
        valid_entity = self.entity.check_traits(self.entity_requirements, self.entity_blacklist)

        if self.target == None:
            valid_target = True
        else:
            valid_target = self.target.check_traits(self.target_requirements, self.target_blacklist)

        if self.extra_condition == None:
            valid_condition = True
        else:
            valid_condition = self.extra_condition(self.entity, self.target)

        return valid_entity and valid_target and valid_condition

    def execution_probability(self, theoretical=False):
        execution_probability = self.base_probability
        
        for approach_name, approach_value in self.approaches.items():
            execution_probability = execution_probability * (1 + self.entity.approach_modifiers[approach_name] * approach_value)


        for cost_name, cost_value in self.costs.items():
            if (self.entity.variables[cost_name] >= cost_value).all():
                execution_probability = execution_probability * (1 - torch.pow(autograd.Variable(torch.FloatTensor([0.5])), self.entity.variables[cost_name] / cost_value))
            else:
                if theoretical:
                    if (self.entity.variables[cost_name] > 1e-10).all():
                        execution_probability = execution_probability * (1 - torch.pow(autograd.Variable(torch.FloatTensor([0.5])), self.entity.variables[cost_name] / cost_value))
                    else:
                        execution_probability = execution_probability * 1e-10 * torch.pow(autograd.Variable(torch.FloatTensor([2.0])), self.entity.variables[cost_name])
                else:
                    return torch.FloatTensor([0.0])

        #execution_probability = torch.clamp(execution_probability, 0.0, 1.0)
        #Clamps while keeping the gradient and the contributions
        if (execution_probability > 1 - 1e-10).all():
            execution_probability = execution_probability - (execution_probability.data.numpy()[0] - (1 - 1e-10))
        elif (execution_probability < 1e-10).all():
            execution_probability = execution_probability - (execution_probability.data.numpy()[0] - 1e-10)

        return execution_probability

    def run_theoretical_action(self, execution_probability):
        for cost_name, cost_value in self.costs.items():
            self.entity.variables[cost_name] = self.entity.variables[cost_name] - cost_value * execution_probability
        
        self.action(self.entity, self.target, execution_probability)

    def run_real_action(self):
        #Check costs in case they are no longer satisfied
        for cost_name, cost_value in self.costs.items():
            if (self.entity.variables[cost_name] < cost_value).all():
                return

        for cost_name, cost_value in self.costs.items():
            self.entity.variables[cost_name] -= cost_value

        self.action(self.entity, self.target, 1.0)
        print(self.flavour_text)

    def clip_base_probability(self):
        if (self.base_probability > 1).all():
            self.base_probability.data = torch.FloatTensor([1.0])
        if (self.base_probability < 1e-10).all():
            self.base_probability.data = torch.FloatTensor([1e-10])

    def save_probability(self):
        self.saved_probability = self.base_probability.data.clone()

    def restore_probability(self):
        self.base_probability.data = torch.FloatTensor(self.saved_probability)
        self.saved_probability = None

    def real_optimise(self, learning_rate):
        if (self.base_probability.data > self.saved_probability).all():
            self.base_probability.data = torch.FloatTensor(self.saved_probability + learning_rate)
        elif (self.base_probability.data < self.saved_probability).all():
            self.base_probability.data = torch.FloatTensor(self.saved_probability - learning_rate)

        self.saved_probability = None

    def reset(self):
        self.base_probability = autograd.Variable(torch.FloatTensor(self.base_probability.data.clone()), requires_grad=True)



class EntityTemplate(Template):
    def __init__(self, constructor, requirements, blacklist, extra_condition=None):
        super().__init__(constructor)
        self.requirements = requirements
        self.blacklist = blacklist
        self.extra_condition = extra_condition

    def check(self, zone):
        if not check_traits(zone, self.requirements, self.blacklist):
            return False

        if self.extra_condition == None:
           return True
        else:
            return self.extra_condition(zone)

        return valid_traits and valid_condition

class Entity:
    def __init__(self, name, variables, approach_modifiers, traits, unique_actions, happiness_function, zone=None):
        self.name = name
        self.variables = {}

        for variable_name, variable_value in variables.items():
            self.variables[variable_name] = autograd.Variable(torch.FloatTensor([variable_value]))

        self.approach_modifiers = approach_modifiers
        self.traits = {}

        for trait in traits:
            self.traits[trait.name] = EntityTrait(trait.name, trait.intensity.data, trait.update_action)

        self.unique_actions = unique_actions
        self.happiness_function = happiness_function

        self.actions = list(unique_actions)

        self.relations = {}

        self.zone = zone

        self.saved_variables = None
        self.saved_intensities = None

    def save_state(self):
        self.saved_variables = {}
        for variable_name, variable_value in self.variables.items():
            self.saved_variables[variable_name] = variable_value.data.clone()

        self.saved_intensities = {}

        for name, trait in self.traits.items():
            self.saved_intensities[name] = trait.intensity.data.clone()

    def restore_state(self):
        for variable_name, variable_value in self.saved_variables.items():
            self.variables[variable_name].data = torch.FloatTensor(variable_value)

        self.saved_variables = None

        for name, trait_intensity in self.saved_intensities.items():
            self.traits[name].intensity.data = torch.FloatTensor(trait_intensity)

        self.saved_intensities = None

    def reset(self):
        for variable_name, variable in self.variables.items():
            self.variables[variable_name] = autograd.Variable(torch.FloatTensor(variable.data.clone()))

        for trait_name, trait in self.traits.items():
            self.traits[trait_name].intensity = autograd.Variable(torch.FloatTensor(trait.intensity.data.clone()))

    def check_traits(self, requirements, blacklist):
        trait_names = self.traits.keys()

        for requirement in requirements:
            if requirement not in trait_names:
                return False

        for blacklist_element in blacklist:
            if blacklist_element in trait_names:
                return False

        return True

    def remove_old_events(self):
        old_event_names = []

        for name, event in [(name, event) for name, event in self.traits.items() if isinstance(event, EntityEvent)]:
            if event.stop_condition != None and event.stop_condition(self):
                old_events.append(name)

        for name in old_event_names:
            self.traits.pop(name)

    def add_possible_events(self, event_templates):
        for event_template in event_templates:
            if event_template.check(self):
                event = event_template.make_event()

                if event.name in self.traits.keys():
                    if event.intensity > self.traits[event.name].intensity:
                        self.traits[event.name] = event
                else:
                    self.traits[event.name] = event

    def update_traits(self, time_coefficient=1):
        for name, trait in self.traits.items():
            trait.update(self, time_coefficient)

    def get_happiness(self):
        return self.happiness_function(self)

def initialise(entities, targeted_templates, targetless_templates):
    #Determina le azioni compatibili
    for template in targetless_templates:
        for entity in entities:
            entity.actions.append(template.constructor(entity, None))

    for entity in entities:
        for target in [x for x in entities if x != entity]:

            entity.relations[target.name] = 0

            for template in targeted_templates:
                entity.actions.append(template.constructor(entity, target))

def get_all_actions(root):
    actions = []

    for entity in root.get_all_entities():
        actions += entity.actions

    return actions

def run_isolated_turn(root, entities, event_templates, foresight, time_coefficient, learning_rate, executed_actions, optimisable_actions, simulate_reasoning):
    def simulated_optimisation(entity):
        theoretical_optimisation_actions = list(set(executed_actions).intersection(set(entity.actions)))
        happiness = torch.sum(-entity.get_happiness())
        grads = autograd.grad(happiness, [x.base_probability for x in theoretical_optimisation_actions], create_graph=True, only_inputs=True, allow_unused=True)

        for action, grad in zip(theoretical_optimisation_actions, grads):
            if grad is not None:
                action.base_probability = action.base_probability - learning_rate * grad
    
    #Salva e stacca lo stato pre-teorico
    root.reset()
    root.save_state()
    root.reset()
    for entity in entities:
        entity.reset()
        entity.save_state()
        entity.reset()

    for action in executed_actions:
        action.save_probability()

    #Esegui azioni con probabilità teoriche
    for i in range(foresight):
        random.shuffle(executed_actions)

        for action in executed_actions:
            theoretical_probability = action.execution_probability(theoretical=True)
            action.run_theoretical_action(theoretical_probability * time_coefficient)

        random.shuffle(entities)
        root.update_traits(time_coefficient)

        for entity in entities:
            entity.update_traits(time_coefficient)

        #Simula i ragionamenti dopo le azioni
        if simulate_reasoning:
            for entity in entities:
                simulated_optimisation(entity)

    if not simulate_reasoning:
        for entity in entities:
            simulated_optimisation(entity)

    

    #Recupera lo stato pre-teorico
    root.reset()
    root.restore_state()
    root.reset()
    for entity in entities:
        entity.reset()
        entity.restore_state()
        entity.reset()

    #Ottimizza le azioni che ci interessano
    for action in optimisable_actions:
        action.real_optimise(learning_rate)

    #Recupera le probabilità delle azioni che non devono essere cambiate
    for action in [x for x in executed_actions if x not in optimisable_actions]:
        action.restore_probability()

    for action in executed_actions:
        action.reset()

def run_turn(root, entities, event_templates, foresight, time_coefficient, learning_rate, execute_actions=True, important_actions=[]):
    #TODO: Esegui le azioni normali e poi quelle segrete

    #Problema: le modifiche teoriche alle probabilità hanno effetti veri
    #Soluzione: Dividere le azioni in normali, diplomatiche e segrete
    #Le diplomatiche si ottimizzano in un ambiente isolato

    actions = [x for x in get_all_actions(root) if x.valid()]
    standard_actions = [x for x in actions if x.action_type == 'Standard']
    diplomatic_actions = [x for x in actions if x.action_type == 'Diplomatic']

    #Step 1: Optimise diplomatic actions using public actions
    run_isolated_turn(root, entities, event_templates, foresight, time_coefficient, learning_rate, standard_actions + diplomatic_actions, diplomatic_actions, True)

    #Step 2: Optimise the hidden actions of each entity using standard actions and the entity's hidden actions
    for entity in entities:
        hidden_actions = [x for x in entity.actions if x.action_type == 'Hidden']
        if len(hidden_actions) > 0:
            run_isolated_turn(root, entities, event_templates, foresight, time_coefficient, learning_rate, standard_actions + hidden_actions, hidden_actions, False)

    #Step 3: Optimise the standard actions using standard actions
    run_isolated_turn(root, entities, event_templates, foresight, time_coefficient, learning_rate, standard_actions, standard_actions, False)


    #Rolla le azioni
    random.shuffle(actions)

    if execute_actions:
        for action in actions:
            real_probability = action.execution_probability(theoretical=False)

            if (real_probability > 0).all() and (random.random() < time_coefficient * real_probability).all():
                if action.name in important_actions:
                    answer = input('Request for Action "{}". Empty to approve, n to deny'.format(action.flavour_text))
                    if answer != 'n':
                        action.run_real_action()
                else:
                    action.run_real_action()

    #Aggiusta gli eventi ed esegui i tratti
    for entity in entities:
        entity.remove_old_events()
        entity.add_possible_events(event_templates)
        entity.update_traits(time_coefficient)

    #Correggi le probabilità fuori range
    for action in actions:
        action.clip_base_probability()

def get_relation(root, main_entity, entities, target, foresight, time_coefficient=1):
    standard_happiness = evaluate_happiness(root, main_entity, list(entities), [x for x in entities if x != target], foresight, time_coefficient)
    new_happiness = evaluate_happiness(root, main_entity, list(entities), list(entities), foresight, time_coefficient)

    return new_happiness - standard_happiness

def evaluate_happiness(root, main_entity, entities, active_entities, foresight, time_coefficient=1):
    #Salva e stacca lo stato pre-teorico
    root.reset()
    root.save_state()
    root.reset()
    for entity in entities:
        entity.reset()
        entity.save_state()
        entity.reset()

    #Esegui azioni teoriche con probabilità effettive
    actions = get_all_actions(root)
    
    for i in range(foresight):
        #random.shuffle(actions)

        for action in [x for x in actions if x.valid()]:
            theoretical_probability = action.execution_probability(theoretical=True)
            action.run_theoretical_action(theoretical_probability * time_coefficient)

        #random.shuffle(entities)
        for entity in entities:
            entity.update_traits(time_coefficient)

        root.update_traits(time_coefficient)

    happiness = main_entity.get_happiness().data.clone()

    #Recupera lo stato pre-teorico
    root.reset()
    root.restore_state()
    root.reset()
    for entity in entities:
        entity.reset()
        entity.restore_state()
        entity.reset()

    return happiness

#TODO:
##Gestione informazioni
##Odio non razionale?
##Memoria
#Nota: Le azioni di influenza teoriche possono comunque influenzare altre azioni di influenza

def adjust_core_variable(entity, variable_name, new_value, real):
    difference = torch.abs(entity.variables[variable_name] - new_value)
    entity.variables[variable_name] = new_value

    if not real:
        entity.variables['base_happiness'] = entity.variables['base_happiness'] - difference * entity.variables['autonomy_coefficient']

def main():
    def standard_variables(strength, influence, money, intrigue, learning):
        return {
            'strength' : strength,
            'influence' : influence,
            'money' : money,
            'intrigue' : intrigue,
            'learning' : learning
            }
    def extended_variables(strength, strength_weight, influence, influence_weight, money, money_weight, intrigue, intrigue_weight, learning, learning_weight, autonomy_coefficient=1):
        variables = standard_variables(strength, influence, money, intrigue, learning)
        variables['strength_weight'] = strength_weight
        variables['influence_weight'] = influence_weight
        variables['money_weight'] = money_weight
        variables['intrigue_weight'] = intrigue_weight
        variables['learning_weight'] = learning_weight

        variables['base_happiness'] = 0
        variables['autonomy_coefficient'] = autonomy_coefficient

        return variables
    def standard_happiness(entity):
        happiness = entity.variables['base_happiness'] + \
        entity.variables['strength'] * entity.variables['strength_weight'] + \
        entity.variables['influence'] * entity.variables['influence_weight'] + \
        entity.variables['money'] * entity.variables['money_weight'] + \
        entity.variables['intrigue'] * entity.variables['intrigue_weight'] + \
        entity.variables['learning'] * entity.variables['learning_weight']

        return happiness

    def normalized_happiness(entity):
        happiness = standard_happiness(entity)

        total_weight = entity.variables['strength_weight'] + \
            entity.variables['influence_weight'] + \
            entity.variables['money_weight'] + \
            entity.variables['intrigue_weight'] + \
            entity.variables['learning_weight']

        return happiness / total_weight


    def peasant_happiness(peasant):
        return peasant.variables['money']

    def land_owner_action(entity, intensity):
        entity.variables['strength'] = entity.variables['strength'] + 10 * intensity

    def revolt_action(entity, target, execution_probability):
        target.variables['strength'] = target.variables['strength'] - 5 * execution_probability

    def special_tax_action(entity, target, execution_probability):
        entity.variables['money'] = entity.variables['money'] + 20 * execution_probability
        target.variables['money'] = target.variables['money'] - 20 * execution_probability

    def abolish_feudal_privileges_action(entity, target, execution_probability):
        target.traits['Land Owner'].intensity = target.traits['Land Owner'].intensity * (1 - execution_probability)

    def burn_money_action(entity, target, execution_probability):
        entity.variables['money'] = entity.variables['money'] - 20 * execution_probability

    def polite_request_action(entity, target, execution_probability):
        #taxes_action = [x for x in target.actions if x.target == entity and x.name == 'Special Tax'][0]
        #taxes_action.base_probability = torch.clamp(taxes_action.base_probability - 0.05 * execution_probability, 0, 1)
        real = execution_probability == 1

        if not isinstance(real, bool):
            real = real.all()

        adjust_core_variable(target, 'money_weight', target.variables['money_weight'] * (1 - execution_probability), real)

    def steal_money_action(entity, target, execution_probability):
        entity.variables['money'] = entity.variables['money'] + 20 * execution_probability
        target.variables['money'] = target.variables['money'] - 20 * execution_probability


    minnestrad_kingdom=Zone('Minnestrad', [EntityTrait('Kingdom')],
                            subzones=[
                                Zone('Childrin', [EntityTrait('Duchy')], subzones=[
                                    Zone('Talos', [EntityTrait('County')]),
                                    Zone('Firdal', [EntityTrait('County')]),
                                    Zone('Pataj', [EntityTrait('County')])
                                    ])
                                ])

    land_owner = EntityTrait('Land Owner', update_action=land_owner_action)

    revolt = EntityActionTemplate(lambda entity, target: EntityAction('Revolt', '{Entity} revolts against {Target}!', 
                                  {'military' : 1},
                                  0.05,
                                  {'strength' : 10}, 
                                  revolt_action, entity, target,
                                 target_requirements=['Land Owner']))

    special_tax = EntityActionTemplate(lambda entity, target: EntityAction('Special Tax', '{Entity} collects a special tax from {Target}', 
                               {'military' : 1},
                               0.05,
                               {'strength': 10},
                               special_tax_action, entity, target,
                               entity_requirements=['Land Owner'],
                               target_blacklist=['Land Owner']
                               ))

    steal_money = EntityActionTemplate(lambda entity, target: EntityAction('Steal Money', '{Entity} steals money from {Target}',
                                                                           {},
                                                                           0.05,
                                                                           {'strength': 10},
                                                                           steal_money_action, entity, target,
                                                                           action_type='Hidden'))

    abolish_feudal_privileges = EntityActionTemplate(lambda entity, target: EntityAction('Abolish Feudal Privileges', '{Entity} abolishes the feudal privileges of {Target}',
                                                     {'diplomacy': 1},
                                                     0.05,
                                                     {'strength' : 50},
                                                     abolish_feudal_privileges_action, entity, target,
                                                     target_requirements=['Land Owner']
                                                     ))
    burn_money = EntityActionTemplate(lambda entity, _: EntityAction('Burn Money', '{Entity} is burning money!',
                                      {},
                                      0.1,
                                      {},
                                      burn_money_action, entity))

    polite_request = EntityActionTemplate(lambda entity, target: EntityAction('Polite Request', '{Entity} asks {Target} not to impose taxes',
                                          {'diplomacy': 1},
                                          0.05,
                                          {},
                                          polite_request_action, entity, target,
                                          action_type='Diplomatic',
                                          target_requirements=['Land Owner']))

    entity_templates = [
        EntityTemplate(lambda zone: Entity(zone.name + ' King', 
               extended_variables(100, 0,
                                 50, 0,
                                 2000, 1,
                                 20, 0,
                                 10, 0), 
               {'military' : 0, 'diplomacy' : -0.5, 'intrigue' : -0.8},
               [land_owner],
               [],
               standard_happiness
              ), ['Kingdom'], []),

        EntityTemplate(lambda zone: Entity(zone.name + ' Peasants',
               standard_variables(100, 0, 2000, 0, 0),
               {'military' : 0, 'diplomacy' : 0, 'intrigue' : -0.2},
               [],
               [],
               peasant_happiness), ['County'], [])
        ]

    minnestrad_kingdom.add_possible_entities(entity_templates)
    entities = minnestrad_kingdom.get_all_entities()

    initialise(entities, [revolt, abolish_feudal_privileges, special_tax, polite_request, steal_money], [burn_money])
    #initialise(entities, [special_tax], [])

    time_coefficient = 0.1
    learning_rate = 0.01

    for i in range(150):
        print('Turn {}'.format(i + 1))
        run_turn(minnestrad_kingdom, entities, [], 10, time_coefficient, learning_rate, i > 100, [])

    happiness_time_coefficient = 1

    for entity in sorted(entities, key=lambda x: x.name):
        print('====={}====='.format(entity.name))
        print('Happiness: {}'.format(entity.happiness_function(entity).data.numpy()))
        print('Strength: {}'.format(entity.variables['strength'].data.numpy()))
        for action in [x for x in entity.actions if x.valid()]:
            print('Evaluating ' + action.name)
            print('Base: {}'.format(action.base_probability.data.numpy()))
            #print('Theoretical: {}'.format(action.execution_probability(theoretical=True).data.numpy()))
            real_probability = action.execution_probability(theoretical=True)

            if isinstance(real_probability, autograd.Variable):
                real_probability = real_probability.data

            print('Theoretical: {}'.format(real_probability.numpy()))

        for target in [x for x in entities if x != entity]:
            print("{}'s opinion of {}: {}".format(entity.name, target.name, get_relation(minnestrad_kingdom, entity, entities, target, 10, happiness_time_coefficient)))
    print([x.name for x in minnestrad_kingdom.entities])

if __name__ == '__main__':
    main()
