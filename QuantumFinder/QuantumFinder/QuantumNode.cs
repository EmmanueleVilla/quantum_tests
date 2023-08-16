using System;
using System.Text;

namespace QuantumFinder
{
	public class QuantumNode
	{
        public int[] phase = new int[512];
        public string key;
        public int fitness;
        public QuantumGate operation;

        public QuantumNode(int[] phase, QuantumGate operation, int fitness, string key)
        {
            this.phase = phase;
            this.operation = operation;
            this.fitness = fitness;
            this.key = key;
        }

        public override string ToString()
        {
            return key;
        }
    }
}

