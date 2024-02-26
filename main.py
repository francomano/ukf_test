import ode4
import controller
import ukf

controller = controller.Controller()
ukf = ukf.UKF()

starting_conf = [0,0,0,0,0,0,0]
for i in range(10):

    cmd = controller.command(starting_conf)
    print(cmd)

    #ode4.f() 
    
    #ukf.read_measures() #must be implemented
    #ukf.compute_sigma_points()

    print(ukf.x)
