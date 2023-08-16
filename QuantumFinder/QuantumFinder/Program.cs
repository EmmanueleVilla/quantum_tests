﻿// See https://aka.ms/new-console-template for more information
using System.Numerics;
using System.Text;
using QuantumFinder;

Console.WriteLine("Hello, World!");

var target = "+----+----+-+++--+--++------++---+++-+--+++++++--+---+-----------+++++++--+-+++-++++++++----++--+++++++++++++++-++++++++---------+++++++++++++++-+--++------++--++++++++++++++++-+---+-----------+++++++--+-+++--+--++------++--+++++++++++++++--+---+-----------+++++++++++++++++++++++++++++++-+++-+--+++++++--+---+----------+++++++++++++++++++++++++++++++++++++++++++++++-++++++++---------+++++++++++++++-+--++------++---+++-+--+++++++--+---+-----------+++++++--+-+++--+--++------++---+++-+----+------+---+----------";
Console.WriteLine(target.Length);

var targetPhases = new int[target.Length];
var startingPhase = new int[target.Length];
Array.Fill(startingPhase, 1);
Console.WriteLine("Starting phases: " + String.Join(", ", startingPhase));

var operations = (new Operations3x3().GetQuantumGates()).Concat(new AdditionalOperations3x3().GetQuantumGates());

var visited = new HashSet<String>();
var best = new QuantumNode(startingPhase, new QuantumGate("", startingPhase), 0, phaseToString(startingPhase));
var frontier = new SortedList<int, QuantumNode>(new DuplicateKeyComparer<int>());
frontier.Add(0, best);
var maxFitness = 0;
var nodes_checked = 0;
var exploded = 0;
while(frontier.Count > 0)
{
    if(frontier.Count > 15000)
    {
        var temp = new SortedList<int, QuantumNode>(new DuplicateKeyComparer<int>());
        for(int i=0; i< 10000; i++)
        {
            temp.Add(frontier.GetKeyAtIndex(i), frontier.GetValueAtIndex(i));
        }
        frontier = temp;
    }
    var current = frontier.GetValueAtIndex(0);
    frontier.RemoveAt(0);
    exploded++;

    if (exploded % 5000 == 0)
    {
        Console.WriteLine("nodes exploded: " + exploded);
        Console.WriteLine("nodes checked: " + nodes_checked);
        Console.WriteLine("nodes queued: " + frontier.Count);
        Console.WriteLine("\n\n");
    }

    if(current.fitness == 512)
    {
        File.WriteAllText("log.txt", current.fitness + "\n" + current.operation);
        Console.WriteLine("END OK\n" + String.Join("\n", current.operation));
        break;
    }

    foreach(QuantumGate operation in operations)
    {
        var newPhase = applyGate(current.phase, operation);
        var stringedPhase = phaseToString(newPhase);
        if(visited.Contains(stringedPhase))
        {
            // already visited
            continue;
        }
        visited.Add(stringedPhase);
        var fitness = calculateFitness(stringedPhase, target);
        var newOp = new QuantumGate(
            current.operation.description + " - " + operation.description,
            operation.operation
            );
        var neighbor = new QuantumNode(
            newPhase,
            newOp,
            fitness,
            stringedPhase
            );
        frontier.Add(fitness, neighbor);
        nodes_checked++;

        if (neighbor.fitness > maxFitness)
        {
            maxFitness = fitness;
            Console.WriteLine("\t>new fitness: " + maxFitness);
            Console.WriteLine("\n\n");
            File.WriteAllText("log.txt", neighbor.fitness + "\n" + neighbor.operation);
        }

        if (neighbor.fitness == 512)
        {
            File.WriteAllText("log.txt", neighbor.fitness + "\n" + neighbor.operation);
            Console.WriteLine("END OK\n" + neighbor.operation);
            break;
        }
    }
}

Console.ReadLine();

int calculateFitness(string stringedPhase, string target)
{
    int fitness = 0;
    for(int i = 0; i < stringedPhase.Length; i++) {
        if (stringedPhase[i] == target[i])
        {
            fitness++;
        }
    }
    return fitness;
}

string phaseToString(int[] phase)
{
    var builder = new StringBuilder();
    foreach (int i in phase)
    {
        if (i > 0)
        {
            builder.Append('+');
        }
        else
        {
            builder.Append('-');
        }
    }
    return builder.ToString();
}

int[] applyGate(int[] phase, QuantumGate operation)
{
    var result = new int[phase.Length];

    for(int i = 0; i< phase.Length; i++)
    {
        result[i] = phase[i] * operation.operation[i];
    }

    return result;
}

Console.ReadLine();

