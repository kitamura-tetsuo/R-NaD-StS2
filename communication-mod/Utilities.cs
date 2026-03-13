using Godot;
using System.Collections.Generic;

namespace communication_mod;

public partial class MainFile : Node
{
    private List<T> FindNodesByType<T>(Node root) where T : class
    {
        var results = new List<T>();
        if (root is T t) results.Add(t);
        foreach (var child in root.GetChildren())
        {
            results.AddRange(FindNodesByType<T>(child));
        }
        return results;
    }
}
