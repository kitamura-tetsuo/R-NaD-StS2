using System;
using System.Reflection;
using System.Linq;

class Test {
    static void Main() {
        try {
            Assembly sts2 = Assembly.LoadFrom("/home/ubuntu/.local/share/Steam/steamapps/common/Slay the Spire 2/data_sts2_linuxbsd_x86_64/sts2.dll");
            var playerType = sts2.GetType("MegaCrit.Sts2.Core.Entities.Players.Player");
            Console.WriteLine("Properties of Player:");
            foreach (var prop in playerType.GetProperties()) Console.WriteLine("  " + prop.Name + " : " + prop.PropertyType.Name);
            var modelDb = sts2.GetType("MegaCrit.Sts2.Core.Models.ModelDb");
            Console.WriteLine("Methods of ModelDb:");
            foreach (var method in modelDb.GetMethods()) Console.WriteLine("  " + method.Name);
        } catch (Exception e) {
            Console.WriteLine(e);
        }
    }
}
