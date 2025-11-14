
function accept=metropolis(g1,g2)

rat=exp(g2-g1);
if rat>1
   accept=1;
else
   r=rand;
   if r<rat
      accept=1;
   else
      accept=0;
   end
end
