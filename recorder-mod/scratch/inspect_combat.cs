
using System;
using System.Reflection;
using System.Linq;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Combat;

public class Info {
    public static void Main() {
        PrintType(typeof(CombatRoom));
        PrintType(typeof(CombatState));
        PrintType(typeof(CombatManager));
    }

    static void PrintType(Type type) {
        Console.WriteLine($"--- {type.FullName} ---");
        foreach (var p in type.GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static)) {
            Console.WriteLine($"Prop: {p.Name} ({p.PropertyType})");
        }
        foreach (var f in type.GetFields(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static)) {
            Console.WriteLine($"Field: {f.Name} ({f.FieldType})");
        }
    }
}
