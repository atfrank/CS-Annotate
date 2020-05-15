def getResids(obj, refatom="C1'"):
  """ function to get residues from a pymol object """
  myspace = {'resid': []}
  cmd.iterate("%s and name %s"%(obj,refatom), "resid.append(resi)", space=myspace)
  return myspace['resid']

def getResnames(obj, refatom="C1'"):
  """ function to get resname from a pymol object """
  myspace = {'resname': []}
  cmd.iterate("%s and name %s"%(obj,refatom), "resname.append(resn)", space=myspace)
  return myspace['resname']

def chi_torsion(obj, filename = "chi.txt"):
    import pandas as pd
    """ function to get chi torsion from a pymol object """
    # https://pymolwiki.org/index.php/Get_Dihedral
    # https://x3dna.org/highlights/the-chi-x-torsion-angle-for-pseudouridine
    resids = getResids(obj, refatom="C1'")
    resnames = getResnames(obj, refatom="C1'")
    chi = []
    for resid, resname in zip(resids, resnames):
        atom1 = "%s and resi %s and name O4'"%(obj, resid)
        atom2 = "%s and resi %s and name C1'"%(obj, resid)
        if resname in ['G', 'GUA', 'RG', 'rG','A', 'ADE', 'RA', 'rA']:
            atom3 = "%s and resi %s and name N9"%(obj, resid)
            atom4 = "%s and resi %s and name C4"%(obj, resid)       
        if resname in ['C', 'CYT', 'RC', 'rC','U', 'URA', 'RU', 'rU']:
            atom3 = "%s and resi %s and name N1"%(obj, resid)
            atom4 = "%s and resi %s and name C2"%(obj, resid)                                               
        chi.append(cmd.get_dihedral(atom1,atom2,atom3,atom4,state=0))
    # save to a pandas dataframe
    df = pd.DataFrame({"resi":resids, "resn":resnames, "chi":chi})
    df.to_csv(filename, sep = " ", index = False)
    return(df)
    
